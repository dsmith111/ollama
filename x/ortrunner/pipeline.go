package ortrunner

import (
	"context"
	"log/slog"
	"time"

	"github.com/ollama/ollama/x/ortrunner/oga"
)

// Generate performs streaming token generation for a completion request.
func (r *Runner) Generate(ctx context.Context, req Request) error {
	maxNewTokens := req.Options.MaxTokens
	if maxNewTokens == 0 {
		maxNewTokens = req.Options.NumPredict
	}

	// Count prompt tokens first — needed for max_length calculation
	promptTokens := 0
	if tokens, err := r.tokenizer.Encode(req.Prompt); err == nil {
		promptTokens = len(tokens)
	}

	// ORT GenAI requires a finite max_length for KV-cache allocation.
	// Ollama uses num_predict=-1 to mean "unlimited", but GenAI cannot
	// handle that. Compute a safe finite max_length.
	const defaultCtx = 4096
	numCtx := defaultCtx

	var maxLength int
	if maxNewTokens <= 0 {
		// Unbounded or unset: generate up to full context minus prompt
		maxLength = max(promptTokens+1, numCtx)
	} else {
		maxLength = max(promptTokens+1, min(numCtx, promptTokens+maxNewTokens))
	}

	params, err := oga.NewGeneratorParams(r.model)
	if err != nil {
		return err
	}
	defer params.Close()

	// Set search parameters
	if req.Options.Temperature > 0 {
		if err := params.SetNumber("temperature", float64(req.Options.Temperature)); err != nil {
			slog.Warn("failed to set temperature", "error", err)
		}
		if err := params.SetBool("do_sample", true); err != nil {
			slog.Warn("failed to set do_sample", "error", err)
		}
	}
	if req.Options.TopP > 0 {
		if err := params.SetNumber("top_p", float64(req.Options.TopP)); err != nil {
			slog.Warn("failed to set top_p", "error", err)
		}
	}
	if req.Options.TopK > 0 {
		if err := params.SetNumber("top_k", float64(req.Options.TopK)); err != nil {
			slog.Warn("failed to set top_k", "error", err)
		}
	}

	slog.Debug("ORT GenAI generation parameters",
		"prompt_tokens", promptTokens, "max_new_tokens", maxNewTokens, "max_length", maxLength)

	if err := params.SetNumber("max_length", float64(maxLength)); err != nil {
		slog.Warn("failed to set max_length", "error", err)
	}

	gen, err := oga.NewGenerator(r.model, params)
	if err != nil {
		return err
	}
	defer gen.Close()

	// Append the prompt tokens
	if err := gen.AppendTokenSequencesFromEncoding(r.tokenizer, req.Prompt); err != nil {
		return err
	}

	stream, err := oga.NewTokenStream(r.tokenizer)
	if err != nil {
		return err
	}
	defer stream.Close()

	promptStart := time.Now()

	// Generate first token (includes prompt processing time)
	if gen.IsDone() {
		req.Responses <- CompletionResponse{Done: true}
		return nil
	}
	if err := gen.GenerateNextToken(); err != nil {
		return err
	}
	promptDuration := time.Since(promptStart)

	genStart := time.Now()
	evalCount := 0

	// Process first generated token
	nextTokens, err := gen.GetNextTokens()
	if err != nil {
		return err
	}
	if len(nextTokens) > 0 {
		text, err := stream.Decode(nextTokens[0])
		if err != nil {
			return err
		}
		if text != "" {
			evalCount++
			select {
			case req.Responses <- CompletionResponse{Content: text}:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	// Generate remaining tokens
	for !gen.IsDone() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if err := gen.GenerateNextToken(); err != nil {
			return err
		}

		nextTokens, err := gen.GetNextTokens()
		if err != nil {
			return err
		}

		if len(nextTokens) == 0 {
			continue
		}

		text, err := stream.Decode(nextTokens[0])
		if err != nil {
			return err
		}

		evalCount++

		if text != "" {
			select {
			case req.Responses <- CompletionResponse{Content: text}:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}

	evalDuration := time.Since(genStart)

	// Send final done response with timing stats
	select {
	case req.Responses <- CompletionResponse{
		Done:               true,
		PromptEvalCount:    promptTokens,
		PromptEvalDuration: promptDuration,
		EvalCount:          evalCount,
		EvalDuration:       evalDuration,
	}:
	case <-ctx.Done():
		return ctx.Err()
	}

	return nil
}
