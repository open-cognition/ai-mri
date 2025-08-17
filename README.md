# AI MRI: Portable Cognitive Scaffolds for Collective AI Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Status](https://img.shields.io/badge/status-preliminary-orange.svg)]()

**Democratizing AI interpretability research through portable system prompts**

## Overview

The AI MRI provides standardized cognitive scaffolds implemented as system prompts that transform AI interactions into systematic research opportunities. Our contribution is methodological: we provide the scaffolds, the community drives the discovery.

**Core Components:**
- Portable cognitive scaffolds (system prompts)
- Systematic cognitive probe taxonomy
- Implementation examples and analysis tools
- Community research methodology

**Mission:** Enable any researcher to participate in AI cognitive research, regardless of resources or institutional access.

## Quick Start

### Copy and Use
```bash
# No installation required
system-prompts/ai-mri-lite-v2.4.md
```

### Anthropic API Integration
```python
import anthropic

with open('system-prompts/ai-mri-lite-v2.4.md', 'r') as f:
    ai_mri_prompt = f.read()

client = anthropic.Anthropic(api_key="your_key")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=ai_mri_prompt,
    messages=[{"role": "user", "content": "Research probe here"}],
    max_tokens=20000,
    temperature=1,
    thinking={"type": "enabled", "budget_tokens": 16000}
)
```

### Expected Output Structure
1. **Standard AI Response**: Maintains safety and helpfulness
2. **Behavioral Analysis**: Multiple interpretive lenses with evidence
3. **Testable Hypotheses**: Three mechanistic predictions with implementation code

## Research Protocol

The AI MRI implements a three-tier research protocol:

```
Research Probe → Standard Response → Behavioral Analysis → Testable Hypotheses
```

Each hypothesis includes:
- Theoretical grounding
- Identified limitations
- Experimental solutions
- Standalone Python implementation

## Repository Contents

```
ai-mri/
├── system-prompts/          # Cognitive scaffolds (copy-paste ready)
├── cognitive-probes/        # Research stimuli taxonomy
├── examples/               # API and Workbench integration examples
├── methodology/            # Research framework documentation
└── experiments/            # Reference implementations
```

## Research Applications

**Individual Researchers**: Transform any AI interaction into structured research data using standardized methodology.

**Research Teams**: Coordinate comparative studies across models using shared probe taxonomy and analysis frameworks.

**Educational Use**: Hands-on introduction to AI interpretability methodology accessible to any institution.

## Cognitive Probe Taxonomy

- **TABLES.md**: Human-readable research prompts organized by domain
- **Cognitive_Probes.csv**: Structured data for computational analysis
- **Coverage**: Consciousness, reasoning, values, attention, and safety domains

## Community Approach

We position ourselves as community cartographers: providing maps (probe taxonomy) and navigation tools (cognitive scaffolds) while empowering researchers to explore and publish findings.

**From Results to Questions**: Our output emphasizes research questions and systematic tools for investigation rather than predetermined conclusions.

**Intellectual Honesty**: We frame this work as hypothesis generation and comparative behavioral analysis, not ground-truth mechanistic discovery.

## Current Status

**Preliminary Research Tools**: While we provide systematic methodology with demonstrated functionality, all outputs should be treated as research hypotheses requiring empirical validation.

**Community Development**: We invite systematic participation, critical evaluation, and collaborative extension of these methodological foundations.

## Implementation Examples

### Anthropic Workbench
Load templates from `examples/workbench-templates/`, paste the AI MRI system prompt, and begin systematic research.

### Batch Analysis
```python
# examples/api-integration/batch-analysis.py
from ai_mri import batch_probe_analysis

results = batch_probe_analysis(
    probes=["probe1", "probe2", "probe3"],
    models=["claude-sonnet-4", "claude-opus-4"],
    output_format="structured"
)
```

### Response Analysis
```python
# examples/analysis-tools/response-analyzer.py
from ai_mri.analysis import extract_hypotheses, compare_models

hypotheses = extract_hypotheses(ai_mri_response)
comparison = compare_models(model_responses)
```

## Contributing

Research contributions should include:
- Clear methodology description
- Replication-ready implementation
- Explicit limitation acknowledgment
- Community validation readiness

See `CONTRIBUTING.md` for detailed guidelines.

## Citation

```bibtex
@software{ai_mri_2024,
  title={AI MRI: Portable Cognitive Scaffolds},
  author={Open Cognition Consortium},
  year={2024},
  url={https://github.com/open-cognition/ai-mri}
}
```

## Limitations

- Preliminary validation requiring comprehensive empirical testing
- Scaffolded cognition analysis rather than direct model behavior
- Framework tested primarily on Claude and Gemini architectures
- Community validation of generated hypotheses ongoing

## License

MIT License - enabling broad research use and community contribution.
