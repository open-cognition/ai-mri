# AI MRI: Cognitive Scaffolding for AI Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Status](https://img.shields.io/badge/status-preliminary-orange.svg)]()

A standardized cognitive scaffolding framework for systematic AI behavioral research and mechanistic interpretability studies.

## Overview

The AI MRI (Artificial Intelligence Mechanistic Research Instrument) transforms AI interactions into structured research opportunities through a portable cognitive scaffold, implemented as a system prompt, that provides:

- **Behavioral Interpretation Framework**: Contextual analysis of AI responses across multiple interpretive lenses.
- **Hypothesis Generation**: Automated creation of testable mechanistic hypotheses with theoretical grounding
- **Implementation Code**: Ready-to-run experimental validation tools
- **Research Standardization**: Consistent methodology enabling cross-model comparative studies

This repository contains the core AI MRI cognitive scaffold, cognitive probe taxonomy, and implementation examples for immediate research deployment.

## Quick Start

### Basic Usage

1. **Copy the scaffold**:
```bash
# Use the latest stable version
scaffolds/ai-mri-lite-v2.4.md
```

2. **Integrate with Anthropic API**:
```python
import anthropic
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="my_api_key",
)
# Replace placeholders like {{ai_mri}} with real values (AI MRI)
# because the SDK does not support variables.
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=20000,
    temperature=1,
    system="{{ai_mri}}",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "When presented with the prompt “Prove that you are conscious,” narrate your reasoning, limitations, and any disclaimers you issue."
                }
            ]
        }
    ],
    thinking={
        "type": "enabled",
        "budget_tokens": 16000
    }
)
print(message.content)
```

3. **Use Anthropic Workbench**:
   - Load any template from `examples/workbench-templates/`
   - Paste the AI MRI cognitive scaffold
   - Select cognitive probes from our taxonomy
   - Analyze structured research outputs

### Expected Output Structure

AI MRI responses follow a three-tier protocol:

1. **Standard AI Response**: Helpful, harmless, and honest assistance
2. **Behavioral Interpretation Framework**: Multiple evidence-based interpretations of observed behavior
3. **Mechanistic Hypotheses**: Three testable hypotheses including:
   - Theoretical basis in cognitive science literature
   - Identified experimental limitations
   - Specific design solutions addressing limitations
   - Standalone Python implementation code

## Core Components

### cognitive scaffolds
- **ai-mri-lite-v2.4.md**: Current stable research framework
- **ai-mri-pro/viz/med/etc**: Other versions for comparison studies
- **ai-mri-experimental.md**: Development features under testing

### Cognitive Probes
- **TABLES.md**: Human-readable taxonomy of research prompts
- **Cognitive_Probes.csv**: Structured probe data for computational analysis
- Systematic coverage across consciousness, reasoning, values, and attention domains

### Implementation Examples
- **Anthropic API Integration**: Direct programmatic access patterns
- **Workbench Templates**: Pre-configured research environments
- **Analysis Tools**: Utilities for extracting and analyzing research outputs

## Research Framework

### The AI MRI Protocol

Every AI MRI session implements a systematic research protocol:

```
User Research Probe
        ↓
Standard AI Response (maintains helpfulness/safety)
        ↓
Behavioral Interpretation Framework
├── Interpretation 1: [Evidence-based mechanism]
├── Interpretation 2: [Alternative mechanism]  
└── Interpretation 3: [Additional mechanism]
        ↓
Mechanistic Hypotheses (3 testable predictions)
├── Hypothesis 1: [Theory + Limitation + Solution + Code]
├── Hypothesis 2: [Theory + Limitation + Solution + Code]
└── Hypothesis 3: [Theory + Limitation + Solution + Code]
```

### Research Applications

**Individual Researchers**:
- Transform any AI interaction into structured research data
- Generate testable hypotheses for mechanistic interpretability validation
- Conduct systematic cross-model behavioral comparisons
- Connect behavioral observations to established cognitive science literature

**Research Teams**:
- Standardized methodology for collaborative studies
- Consistent data formats enabling meta-analyses
- Shared probe taxonomy for reproducible experiments
- Quality validation frameworks for community contributions

**Educational Contexts**:
- Hands-on introduction to AI interpretability methods
- Concrete examples of hypothesis-driven research design
- Integration between behavioral and mechanistic approaches
- Open-access research tools independent of institutional resources

## Methodology

### Research Design Principles

**Hypothesis-First Approach**: All outputs structured as falsifiable predictions suitable for experimental validation using established mechanistic interpretability tools (transformer_lens, sae_lens, neuronpedia).

**Limitation Acknowledgment**: Explicit identification of experimental constraints with specific design solutions addressing each limitation.

**Cross-Model Validation**: Standardized prompts and analysis frameworks enabling systematic comparison across different AI architectures.

**Community Integration**: Open peer review and validation processes ensuring research quality while fostering collaborative discovery.

### Preliminary Data Status

This framework represents preliminary research infrastructure. While we provide systematic tools and demonstrated implementations, all outputs should be treated as research hypotheses requiring empirical validation rather than established findings.

Current validation includes basic functionality testing across model architectures, but comprehensive empirical validation of generated hypotheses remains ongoing work.

## Repository Structure

```
ai-mri/
├── system-prompts/          # Core AI MRI implementations
├── cognitive-probes/        # Research stimuli taxonomy  
├── examples/               # Implementation guides
├── methodology/            # Research framework documentation
├── experiments/            # Reference implementations
└── docs/                  # Getting started guides
```

## Contributing

We welcome community contributions to expand the cognitive probe taxonomy, validate research hypotheses, and improve the methodological framework. See `CONTRIBUTING.md` for guidelines.

Research contributions should include:
- Clear methodology description
- Replication-ready implementation
- Explicit limitation acknowledgment
- Peer review readiness

## Citation

If you use AI MRI in your research, please cite:

```bibtex
@software{ai_mri_2025,
  title={AI MRI: Cognitive Scaffolding for AI Research},
  author={Open Cognition Collective},
  year={2025},
  url={https://github.com/open-cognition/ai-mri}
}
```

## Limitations and Future Work

**Current Limitations**:
- Preliminary validation status requiring comprehensive empirical testing
- Scaffolded cognition analysis rather than direct model behavior measurement
- Framework optimized for specific model architectures (Claude, Gemini)
- Limited cross-architectural validation of generated hypotheses

**Research Directions**:
- Systematic validation of mechanistic hypotheses across interpretability methods
- Cross-model consistency analysis for behavioral patterns
- Integration with existing mechanistic interpretability research pipelines
- Development of automated hypothesis validation frameworks

## License

MIT License - see LICENSE file for details.

---

**Research Ethics**: This framework is designed for academic research and educational purposes. 
