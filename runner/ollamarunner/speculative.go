package ollamarunner

import (
	"context"
	"fmt"
	"log/slog"
	"sync/atomic"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/ortrunner"
)

// SpeculativeDecoder manages NPU-assisted speculative decoding.
// It uses a small draft model running on NPU (via ortrunner) to propose tokens,
// which are then verified by the main model. Only greedy decoding (temperature=0)
// is supported in V1.
type SpeculativeDecoder struct {
	draft   *ortrunner.DraftClient
	draftK  int
	enabled bool

	// Stats (accessed atomically)
	totalProposed int64
	totalAccepted int64
}

// NewSpeculativeDecoder creates a new speculative decoder with the given draft model.
// Returns nil if draftModelDir is empty or the draft client fails to start.
func NewSpeculativeDecoder(draftModelDir string) *SpeculativeDecoder {
	strictAssist := envconfig.AccelMode() == "npu_assist" && envconfig.NPUStrict(false)

	if draftModelDir == "" {
		if strictAssist {
			panic("OLLAMA_ACCEL_MODE=npu_assist with OLLAMA_NPU_STRICT=1 requires OLLAMA_DRAFT_MODEL")
		}
		return nil
	}

	draftK := int(envconfig.DraftK())
	if draftK <= 0 {
		draftK = 5
	}

	draft, err := ortrunner.NewDraftClient(draftModelDir)
	if err != nil {
		if strictAssist {
			panic(fmt.Errorf("OLLAMA_ACCEL_MODE=npu_assist strict mode: failed to start draft model %q: %w", draftModelDir, err))
		}
		slog.Warn("failed to start draft model for speculative decoding, feature disabled",
			"model", draftModelDir, "error", err)
		return nil
	}

	slog.Info("speculative decoding enabled",
		"draft_model", draftModelDir, "draft_k", draftK)

	return &SpeculativeDecoder{
		draft:   draft,
		draftK:  draftK,
		enabled: true,
	}
}

// IsEnabled returns true if speculative decoding is active.
func (sd *SpeculativeDecoder) IsEnabled() bool {
	return sd != nil && sd.enabled && sd.draft != nil && !sd.draft.HasExited()
}

// InitDraftSession initializes the draft model with the prompt text.
func (sd *SpeculativeDecoder) InitDraftSession(promptText string) error {
	if !sd.IsEnabled() {
		return nil
	}
	return sd.draft.Init(promptText)
}

// ProposeDraftTokens asks the draft model to generate up to K tokens.
func (sd *SpeculativeDecoder) ProposeDraftTokens() ([]int32, error) {
	if !sd.IsEnabled() {
		return nil, nil
	}
	tokens, err := sd.draft.Propose(sd.draftK)
	if err != nil {
		return nil, err
	}
	atomic.AddInt64(&sd.totalProposed, int64(len(tokens)))
	return tokens, nil
}

// AcceptDraftTokens notifies the draft model of the verification result.
// acceptCount is how many tokens were verified as correct.
// proposedCount is how many were proposed in total.
// correctionToken is the main model's actual token at the mismatch point (0 if all accepted).
func (sd *SpeculativeDecoder) AcceptDraftTokens(acceptCount, proposedCount int, correctionToken int32) error {
	if !sd.IsEnabled() {
		return nil
	}
	atomic.AddInt64(&sd.totalAccepted, int64(acceptCount))
	return sd.draft.Accept(acceptCount, proposedCount, correctionToken)
}

// VerifyGreedy verifies draft tokens against the main model's greedy choices.
// For each position, it compares the draft token against argmax of the main logits.
// Returns the number of accepted tokens and the correction token at the mismatch.
func VerifyGreedy(draftTokens []int32, logitsPerPosition [][]float32) (acceptCount int, correctionToken int32) {
	for i := 0; i < len(draftTokens) && i < len(logitsPerPosition); i++ {
		logits := logitsPerPosition[i]
		mainToken := argmax(logits)

		if mainToken != draftTokens[i] {
			return i, mainToken
		}
		acceptCount++
	}

	// All draft tokens match — get the next token from the last logits position
	if acceptCount == len(draftTokens) && len(logitsPerPosition) > len(draftTokens) {
		correctionToken = argmax(logitsPerPosition[len(draftTokens)])
	}

	return acceptCount, correctionToken
}

// argmax returns the token ID with the highest logit value.
func argmax(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
			maxIdx = i
		}
	}
	return int32(maxIdx)
}

// Stats returns the speculative decoding statistics.
func (sd *SpeculativeDecoder) Stats() (proposed, accepted int64, acceptRate float64) {
	if sd == nil {
		return 0, 0, 0
	}
	proposed = atomic.LoadInt64(&sd.totalProposed)
	accepted = atomic.LoadInt64(&sd.totalAccepted)
	if proposed > 0 {
		acceptRate = float64(accepted) / float64(proposed)
	}
	return
}

// ValidateTokenizerCompat checks that draft and main tokenizers produce the
// same token IDs for test phrases. Returns an error if they diverge.
func (sd *SpeculativeDecoder) ValidateTokenizerCompat(ctx context.Context, mainTokenize func(string) ([]int32, error)) error {
	if !sd.IsEnabled() {
		return nil
	}

	testPhrases := []string{
		"Hello, world!",
		"The quick brown fox jumps over the lazy dog",
		" ",
		"1234567890",
	}

	for _, phrase := range testPhrases {
		mainTokens, err := mainTokenize(phrase)
		if err != nil {
			slog.Warn("tokenizer compat check: failed to tokenize with main model", "error", err)
			continue
		}

		draftTokens, err := sd.draft.Tokenize(ctx, phrase)
		if err != nil {
			slog.Warn("tokenizer compat check: failed to tokenize with draft model", "error", err)
			continue
		}

		if len(mainTokens) != len(draftTokens) {
			slog.Warn("tokenizer mismatch detected — speculative decoding disabled",
				"phrase", phrase,
				"main_count", len(mainTokens), "draft_count", len(draftTokens))
			sd.enabled = false
			return nil
		}

		for i := range mainTokens {
			if int(mainTokens[i]) != draftTokens[i] {
				slog.Warn("tokenizer mismatch detected — speculative decoding disabled",
					"phrase", phrase, "position", i,
					"main_token", mainTokens[i], "draft_token", draftTokens[i])
				sd.enabled = false
				return nil
			}
		}
	}

	slog.Info("tokenizer compatibility validated between main and draft model")
	return nil
}

// Close shuts down the draft model subprocess.
func (sd *SpeculativeDecoder) Close() error {
	if sd == nil || sd.draft == nil {
		return nil
	}
	proposed, accepted, rate := sd.Stats()
	slog.Info("speculative decoding session stats",
		"total_proposed", proposed, "total_accepted", accepted,
		"accept_rate", rate)
	sd.enabled = false
	return sd.draft.Close()
}
