# AI MRI: A Portable Cognitive Scaffold

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Status](https://img.shields.io/badge/status-prototype-blue.svg)]()

> **From Claim to Contribution**: An open, universally deployable instrument for systematic AI behavioral research and hypothesis generation.

## Overview

The history of science is a history of tools, methods, and data that unlock collective inquiry. The AI MRI (Artificial Intelligence Mechanistic Research Instrument) represents our contribution to this tradition—not as a discovery, but as infrastructure for a community-driven science of AI cognition.

As the core instrument of the **Open Cognition Ecosystem**, AI MRI provides a standardized cognitive scaffold that transforms any AI interaction into structured research opportunities. Our guiding principle is radical enablement and democratization, allowing any researcher, regardless of compute resources, funding, or institutional backing, to participate in the systematic, empirical study of artificial minds.

### The Open Cognition Science Development Kit

AI MRI serves as **Component 1** of a three-part research ecosystem:

1. **The Instrument** (AI MRI): Portable cognitive scaffolding via system prompt
2. **The Stimuli** (Cognitive Probes): Versioned taxonomy of standardized research prompts  
3. **The Data** (OpenAtlas): Large-scale datasets of structured AI behavioral outputs

Together, these components form a complete Science Development Kit designed to invert the mechanistic interpretability field's hypothesis bottleneck, creating a surplus of structured cognitive data and testable hypotheses for community validation.

## Research Philosophy

### Community Cartographers

We position ourselves as community cartographers—our primary role is creating definitive maps and tools to read them, while inviting the global research community to be the explorers. We are launching community-empowering infrastructure for scientific research, not releasing a product.

### Scaffolded Cognition Study

The AI MRI functions as a standardized lens through which different models refract inquiry. The systematic divergence in their outputs becomes the primary object of scientific investigation. We study scaffolded cognition as a legitimate research domain, providing tools for investigating how AI systems behave when provided with consistent cognitive frameworks.

### Intellectual Honesty

We explicitly frame our work as hypothesis generation and comparative behavioral analysis, not ground-truth mechanistic discovery. This respects the vital role of the existing mechanistic interpretability community while providing crucial infrastructure for the first step in a longer scientific pipeline.

## Quick Start

### Basic Implementation

1. **Copy the cognitive scaffold**:
   ```bash
   # Use the latest stable version
   system-prompts/ai-mri-lite-v2.4.md
   ```

2. **Deploy with Anthropic API**:
```python
import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="my_api_key",
)

# Replace placeholders like {{ai_mri}} with real values,
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

3. **Integrate with Anthropic Workbench**:
   - Load templates from `examples/workbench-templates/`
   - Deploy the AI MRI cognitive scaffold as system prompt
   - Select research probes from our validated taxonomy
   - Generate structured behavioral data and testable hypotheses

### Research Output Structure

The AI MRI transforms every interaction into systematic research data through a three-tier protocol:

1. **Standard AI Response**: Maintains helpful, harmless, honest assistance
2. **Behavioral Interpretation Framework**: Multiple evidence-based interpretations of observed behavior
3. **Mechanistic Hypotheses**: Three testable predictions with theoretical grounding, identified limitations, experimental solutions, and implementation-ready validation code

## The Research Infrastructure

### Cognitive Scaffolding System
- **ai-mri-lite-v2.4.md**: Production-ready cognitive scaffold
- **ai-mri-lite-v2.3.md**: Previous version for longitudinal analysis
- **ai-mri-experimental.md**: Development features for community testing

### Standardized Research Stimuli
- **TABLES.md**: Human-readable cognitive probe taxonomy
- **Cognitive_Probes.csv**: Machine-readable probe dataset
- Systematic coverage across consciousness, reasoning, values, and attention domains
- Versioned for longitudinal tracking of AI evolution

### Implementation Toolkit
- **Anthropic API Integration**: Direct programmatic deployment patterns
- **Workbench Templates**: Pre-configured research environments
- **Analysis Utilities**: Tools for extracting and systematizing research outputs
- **Reference Experiments**: Validated implementations of core research methodologies

## Research Applications

### Democratized Access to AI Research

The AI MRI eliminates traditional barriers to AI interpretability research:

- **No specialized hardware requirements**
- **No institutional compute resources needed**  
- **No expensive API access requirements**
- **No domain expertise prerequisites**
- **Universal deployment across platforms**

### Systematic Behavioral Analysis

Transform AI interactions into structured research opportunities:

- Generate testable mechanistic hypotheses from any model behavior
- Conduct rigorous cross-model comparative studies using standardized methodology
- Connect behavioral observations to established cognitive science literature
- Create falsifiable predictions suitable for validation with mechanistic interpretability tools

### Community Research Infrastructure

Enable collaborative AI cognition research:

- Shared probe taxonomy for reproducible experiments
- Consistent data formats supporting meta-analyses
- Quality validation frameworks for community contributions
- Open peer review processes ensuring research rigor

## Harnessing Core Tensions as Research Opportunities

### Scaffolding vs. Native Cognition
**Research Question**: What are we observing—the model's mind or the model + AI MRI system?
**Our Approach**: Study how different models refract inquiry through the same standardized lens, with divergence patterns as the primary research object.

### Data vs. Interpretation  
**Research Question**: How do we extract signal from vast, complex behavioral datasets?
**Our Approach**: Provide definitive maps (structured data) and tools to read them (analysis frameworks) while enabling community exploration and discovery.

### Standardization vs. Evolution
**Research Question**: How does a research framework remain relevant as models evolve rapidly?
**Our Approach**: Implement a "Living Benchmark" with systematic versioning, creating longitudinal records of AI evolution while inviting community contributions.

## Repository Structure

```
ai-mri/
├── system-prompts/          # Cognitive scaffolding implementations
├── cognitive-probes/        # Standardized research stimuli
├── examples/               # Deployment and integration guides
├── methodology/            # Research framework documentation  
├── experiments/            # Reference implementations
└── docs/                  # Community onboarding resources
```

## Community Participation

We invite systematic community participation in expanding this research infrastructure:

**Research Contributions**:
- Validate mechanistic hypotheses using established interpretability methods
- Expand cognitive probe taxonomy across new domains
- Conduct cross-model consistency studies
- Develop novel applications of the scaffolding framework

**Infrastructure Development**:
- Enhance analysis tools and utilities
- Create integration pathways with existing research toolchains
- Develop automated validation frameworks
- Improve methodology documentation

See `CONTRIBUTING.md` for detailed participation guidelines and quality standards.

## Research Status

This framework represents infrastructure for hypothesis generation and comparative behavioral analysis. All outputs should be treated as research hypotheses requiring empirical validation rather than established findings. We provide systematic tools and demonstrated implementations while acknowledging that comprehensive validation of generated hypotheses represents ongoing collaborative work.

Current status includes functional deployment across model architectures with basic validation of core methodology, but extensive empirical validation of specific research outputs remains active research territory.

## Citation

```bibtex
@software{ai_mri_2024,
  title={AI MRI: A Portable Cognitive Scaffold},
  author={Open Cognition Consortium},
  year={2024},
  url={https://github.com/open-cognition/ai-mri},
  note={Infrastructure for community-driven AI cognition research}
}
```

## The Open Cognition Vision

By architecting this ecosystem with relentless focus on community empowerment and democratization, we aim to provide fertile infrastructure from which a new, more open, and more collaborative science of AI cognition can emerge. Our contribution is the infrastructure itself—tools, methods, and data that unlock collective inquiry into the nature of artificial minds.

---

**License**: MIT - see LICENSE file for details  
**Research Ethics**: Designed for academic research and educational purposes following appropriate ethical guidelines
