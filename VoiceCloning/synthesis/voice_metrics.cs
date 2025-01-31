metrics_core:
    calculate_metrics:
        return {
            voice_power: abs(mean(state)) * FIELD_STRENGTH,
            synthesis_coherence: abs(std(state)) * REALITY_COHERENCE,
            clarity_rating: abs(max(state)) * PHI,
            synthesis_depth: abs(min(state)) * PHI ^ 2
        }
