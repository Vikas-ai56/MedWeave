# Global Development Rules — MedWeave

These rules apply to every contributor and AI coding agent working on MedWeave.

## 1. Source of Truth

Before making changes, read:

- `SPEC.md` — overall project requirements and architecture.
- `GLOBAL_RULES.md` — development rules.
- `INTERFACES.md` — shared module interfaces.

Follow these documents instead of independently redesigning the architecture.

If two documents conflict, follow `SPEC.md` and document the conflict for the team to resolve. Do not silently make architectural decisions.

## 2. Project Scope

MedWeave is an academic proof-of-concept for collaborative AI training using U-shaped Split Learning.

The initial target is:
- Four simulated institutions.
- One shared Backbone model.
- Institution-specific Head and Tail models.
- Sequential institutional training.
- External compute peers.
- Configurable activation noise.
- A working end-to-end training prototype.

Do not introduce blockchain, cryptocurrency, complex swarm consensus, clinical integrations, or production infrastructure unless explicitly required by the specification.

## 3. Coding Standards

- Use Python and PyTorch for machine learning components.
- Write readable, modular and maintainable code.
- Use clear names for classes, functions and variables.
- Add type hints to public functions where practical.
- Add docstrings to important classes and functions.
- Avoid unnecessary abstractions and overengineering.
- Do not duplicate existing functionality.
- Handle errors explicitly rather than silently ignoring them.
- Use configuration values instead of hardcoding parameters unnecessarily.

## 4. Module Ownership

Each contributor should work within their assigned module.

Do not modify another member's module or shared interfaces without a valid reason.

If another module requires functionality that does not exist:
1. Check `INTERFACES.md`.
2. Document what is missing.
3. Propose the smallest necessary interface change.
4. Do not silently introduce incompatible changes.

## 5. Dependencies

- Prefer existing project dependencies.
- Do not introduce a new dependency when standard Python or PyTorch functionality is sufficient.
- Document every additional dependency in the appropriate requirements file.
- Avoid unnecessary frameworks.

## 6. Testing

Every module must include relevant tests.

At minimum:
- Test important functions independently.
- Test expected tensor shapes and data types.
- Test error handling for invalid inputs.
- Test model parameter updates during training.
- Test integration against the agreed interfaces.

Do not claim that functionality works unless it has actually been tested.

## 7. Git Practices

- Work on a dedicated feature branch.
- Do not commit directly to `main`.
- Keep commits focused and meaningful.
- Do not overwrite another contributor's work.
- Do not commit datasets, model checkpoints, credentials, or generated files unnecessarily.

## 8. Documentation

Each module should have:
- A short README explaining its purpose.
- Setup and execution instructions.
- Public interfaces and usage examples.
- Relevant tests.
- Known limitations.

Keep documentation practical and concise.

## 9. AI Coding Agent Rules

AI agents must:
- Inspect the existing repository before changing files.
- Read the project specification and shared interfaces.
- Reuse existing code where appropriate.
- Avoid replacing working code without justification.
- Never invent successful test results.
- Report files changed, tests executed and remaining issues.
- Ask for clarification only when a genuinely blocking architectural decision is missing.

## 10. Definition of Done

A module is complete when:
- Its required functionality is implemented.
- Its tests pass.
- It follows the shared interfaces.
- It can be integrated with the other modules.
- Its setup and usage are documented.

The priority is a working, understandable prototype—not unnecessary complexity or excessive abstraction.