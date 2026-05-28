"""Online RL training for V-LEAD pilot networks.

Env lives in `vlead_flight.env` (Gymnasium-compatible). This package owns
the algorithm-side pieces: actor-critic feature extractor reusing the BC
encoder, BC checkpoint -> RL actor warm-start, and SAC / PPO trainers.
"""
