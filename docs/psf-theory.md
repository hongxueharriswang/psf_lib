
# Theoretical Foundations

## The Five Persistence States

| State | Label | Description |
|---|---|---|
| S₁ | Registered | Enrolled but not yet progressing. |
| S₂ | Progressing | Actively completing assessments. |
| S₃ | Stopped-Out | Temporarily inactive with probability of return. |
| S₄ | Withdrawn | Effectively exited with no intent to return. |
| S₅ | Completed | Course requirements fulfilled. |

## Intent to Return

The latent variable Iₜ ∈ {0,1} captures the student's commitment to continue.
It is inferred from proxies such as extension requests, tutor contacts, and
future registrations.

## The Hidden Markov Model

The PSF is implemented as an HMM over six joint states:

1. S₁ (Registered)
2. S₂ (Progressing)
3. S₃ with I=0 (Stopped-Out, no intent)
4. S₃ with I=1 (Stopped-Out, intent)
5. S₄ (Withdrawn)
6. S₅ (Completed)

The joint likelihood factorizes as:
