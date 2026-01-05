# CNNeoPP: A Novel Consensus Framework for Neoantigen Prioritization

## Overview

CNNeoPP (CNNeo Pipeline) is an end-to-end pipeline that derives peptide–HLA candidates from raw sequencing data and prioritizes neoantigens by integrating CNNeo (CNN and NLP-based Neoantigen Prediction Model) immunogenicity predictions using a rank-level consensus strategy. The computational environment is fully reproducible via `environment.yml`.

Terminology note: CNNeoPP refers to the overall neoantigen classification pipeline described in this repository, while CNNeo denotes the core prediction module built into CNNeoPP.

## Repository Contents

This repository contains:

- `models/`: Contains Jupyter notebooks implementing the CNNeo immunogenicity prediction models, including model architecture, training, and evaluation workflows.

- `data/example/`: Includes example input and output files to help you get started with CNNeoPP.

- `docs/`: Detailed documentation on the CNNeoPP workflow and how to use it.

---

## Environment and Installation

This repository uses **`environment.yml`** as the primary (and recommended) way to reproduce the runtime environment.

### Prerequisites

- Install **Miniconda** or **Anaconda** (Conda is required to use `environment.yml`).
- Recommended: use the same OS family as the environment was exported on.  
  The provided `environment.yml` was exported from a Windows Anaconda environment and may require small adjustments on other platforms.

### Create the environment (recommended)

From the repository root (where `environment.yml` is located):

```bash
# Create a new environment named "cnneopp" from environment.yml
# (Recommended to avoid conflicts if the yml has name: base)
conda env create -f environment.yml -n cnneopp

conda activate cnneopp
```

If you already created the environment once and want to update it:

```bash
conda env update -n cnneopp -f environment.yml --prune
conda activate cnneopp
```

### Verify installation

```bash
python --version
python -c "import torch, transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__)"
```

### Optional: enable Jupyter kernel

If you run notebooks, register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name cnneopp --display-name "Python (cnneopp)"
```

### GPU/CPU note (PyTorch)

The exported `environment.yml` pins PyTorch wheels that include a CUDA build (`+cu121`). If you are using a CPU-only machine or a different CUDA setup, environment creation may fail during the pip stage, or PyTorch may not run as expected.

A robust approach is:

1. Create the environment from `environment.yml`.
2. If needed, reinstall PyTorch to match your hardware.

Example (CPU-only):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Example (CUDA 12.1):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Troubleshooting

If you encounter issues during environment setup or package installation, the following tips may help resolve common problems.

**1. Conda environment creation fails**

- Update Conda:
  ```bash
  conda update -n base -c defaults conda
  ```
- On non-Windows systems, try creating the environment using the name specified inside environment.yml.

**2. PyTorch installation errors on CPU-only machines**

- Reinstall PyTorch using the CPU-only wheels as shown above.

**3. CUDA version mismatch**

- Check your local CUDA version:

  ```bash
  nvcc --version
  ```
  
- Reinstall PyTorch with a CUDA build compatible with your system.

**4. Package conflicts on non-Windows operating systems**

- The environment was exported from Windows and may require minor dependency adjustments on Linux or macOS.

- If conflicts occur, consider installing problematic packages individually after environment creation.

If issues persist, please open a GitHub issue and include your operating system, Python version, and error messages.

---

## Input Data Format

CNNeoPP operates on a tabular input file (**CSV** or **TSV**) containing neoantigen candidate information:

Required columns:

- `peptide` — The amino acid sequence of the candidate neoantigen  
- `hla` — The associated HLA allele (e.g., `HLA-A*02:01`)

Optional columns (if available):

- Proteasomal cleavage, TAP transport efficiency, NetCTLpan score, Peptide–HLA binding affinity, Peptide–HLA binding stability, Peptide hydrophobicity, Peptide weight, Peptide entropy, IEDB immunogenicity score, Agretopicity, T-cell contact residues hydrophobicity

Example input file:

- `data/example/input_features.csv`

---

## Running CNNeoPP (End-to-End Workflow Reference)

CNNeoPP is executed in a stepwise manner, combining CNNeo immunogenicity prediction notebooks with downstream rank-level consensus aggregation.
The current implementation is notebook-based and provides a transparent reference workflow for neoantigen prioritization.

### Step 1: CNNeo immunogenicity prediction (submodel inference)
Immunogenicity prediction is performed using the CNNeo submodel notebooks provided in the `models/` directory.
Each notebook independently loads the input feature table, performs feature construction and model inference, and outputs immunogenicity scores for peptide–HLA pairs.

The following CNNeo submodel notebooks are provided:

- `CNNeo_CNN_BioBERT.ipynb`
- `CNNeo_FCNN_BioBERT.ipynb`
- `CNNeo_FCNN_TF.ipynb`

Example (Jupyter-based execution):

```bash
jupyter notebook models/CNNeo_CNN_BioBERT.ipynb
```
Run the remaining CNNeo submodel notebooks in the same way to obtain submodel-level immunogenicity predictions.

### Step 2: Collection of submodel prediction results

After executing the CNNeo submodel notebooks, collect the predicted immunogenicity scores generated by each model into a unified table.

These outputs serve as the input for downstream rank-level consensus aggregation.

Each row typically corresponds to a peptide–HLA pair and includes the predicted label and immunogenicity score produced by a specific submodel.

### Step 3: Rank-level consensus aggregation and final prioritization

CNNeoPP performs rank-level consensus aggregation on the collected submodel outputs to generate the final prioritized list of peptide–HLA pairs.

This step includes:

1. Selecting top-ranked candidates from each submodel

2. Identifying peptides supported by at least two submodels

3. Promoting consensus-supported peptides to the top of the final ranking

4. Appending remaining candidates based on their individual model-specific ranks or scores

The resulting output is a ranked list of candidate neoantigens suitable for downstream experimental validation.

### Notes

CNNeoPP is currently provided as a reference workflow based on Jupyter notebooks, enabling transparency and reproducibility of each modeling step.

An automated command-line pipeline is not required to reproduce the reported results.

Detailed upstream sequencing preprocessing and variant calling commands (e.g., FastQC, alignment, and somatic variant calling) are provided separately in Appendix A for advanced users.

---

## Output Description

The primary output is a ranked list of peptide–HLA pairs, typically including the following fields:

- `Hla Type` — HLA allele associated with the peptide (e.g., HLA-A*02:01)
- `Mutated Peptide` — Amino-acid sequence of the mutated peptide
- `Model` Predictive model(s) supporting the peptide
- `Predicted Label` — Binary prediction label (1 = positive, 0 = negative)
- `Immunogenicity Score` — Predicted immunogenicity score
- `Final Rank` — Final ranking after CNNeoPP prioritization

Example output file:

- `data/example/example_output.csv`

---

## Rank-Level Consensus Logic

The final CNNeoPP ranking is generated by:

1. Selecting the Top-N candidates from each submodel.  
2. Identifying peptides that are supported by at least two models.  
3. Prioritizing peptides with consistent support across models.
4. Appending the remaining peptides based on individual model scores.

---

## Reproducibility and Versioning

- Use the committed `environment.yml` to recreate the environment: `conda env create -f environment.yml -n cnneopp`.
- Record the following when reporting results: OS, Python version, CUDA availability (if applicable), and the exact Git commit/tag.
- For strict reproduction of published results, use a tagged release corresponding to the manuscript version.

---

## CNNeoPP Versioning

To ensure reproducibility and clarity, CNNeo and CNNeoPP are versioned using Semantic Versioning.

CNNeo Version: v1.0.0

CNNeoPP Version: v1.0.0

Each new version of CNNeoPP will be tagged with a version number to reflect the changes made.

## Citation and License

Please cite CNNeoPP as:

> Yu Cai, Rui Chen, Mingming Song, Lei Wang, Zirong Huo, Dongyan Yang, Sitong Zhang, Shenghan Gao, Seungyong Hwang, Ling Bai, Yonggang Lv, Yali Cui, Xi Zhang. **CNNeoPP: A large language model-enhanced deep learning pipeline for personalized neoantigen prediction and liquid biopsy applications**.

License:

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact and GitHub Issues

For questions, issues, or contributions, please open an issue in the GitHub repository. We welcome contributions and bug reports!

Contact:

201720908@stumail.nwu.edu.cn, 202010244@stumail.nwu.edu.cn or xzhangtx@gmail.com

---

# Appendix A: CNNeoPP End-to-End Pipeline Commands

> This appendix provides a reference implementation of a typical upstream bioinformatics workflow for neoantigen discovery.

## 1. Quality control of raw sequencing reads with fastqc

```bash
fastqc input_file.fastq.gz -o output_directory/
```

`input_file.fastq.gz`: Raw sequencing reads in FASTQ format
`output_directory/`: Directory to store quality control reports

This step assesses sequencing quality metrics such as base quality scores, GC content, and adapter contamination.

## 2. Adapter trimming and quality filtering with Trimmomatic-0.39

```bash
java -jar trimmomatic-0.39.jar PE   -phred33   -threads 30   input_R1.fastq.gz input_R2.fastq.gz   output_R1_paired.fastq.gz output_R1_unpaired.fastq.gz   output_R2_paired.fastq.gz output_R2_unpaired.fastq.gz   ILLUMINACLIP:TruSeq3-PE.fa:2:30:10   SLIDINGWINDOW:5:20   LEADING:5   TRAILING:5   MINLEN:50
```

This step removes sequencing adapters and low-quality bases, producing cleaned paired-end reads for downstream analysis.

## 3. Read alignment to the reference genome with BWA

Index the reference genome

```bash
bwa index -a bwtsw Homo_sapiens_assembly38.fasta
```

Align reads

```bash
bwa mem -t 6 -M   -R '@RG\tID:foo_lane\tPL:illumina\tLB:library\tSM:sample_name'   Homo_sapiens_assembly38.fasta.64   output_R1_paired.fastq.gz output_R2_paired.fastq.gz   | samtools view -S -b - > sample.bam
```

This step aligns sequencing reads to the human reference genome (hg38).

## 4. Sorting aligned reads with samtools

```bash
time samtools sort -@ 4 -m 4G -O bam -o sample.sorted.bam sample.bam
```

Sorting is required for efficient downstream processing and variant calling.

## 5. Marking and removing PCR duplicates with Picard

```bash
java -jar picard.jar MarkDuplicates   -REMOVE_DUPLICATES true   -I sample.sorted.bam   -O sample.sorted.markdup.bam   -M sample.markdup_metrics.txt
```

```bash
samtools index sample.sorted.markdup.bam
```

PCR duplicates are marked or removed to reduce false-positive variant calls.

## 6. Base quality score recalibration with GATK4.4

```bash
gatk BaseRecalibrator   -R Homo_sapiens_assembly38.fasta   -I sample.sorted.markdup.bam   --known-sites Homo_sapiens_assembly38.dbsnp138.vcf   --known-sites Mills_and_1000G_gold_standard.indels.hg38.vcf.gz   --known-sites 1000G_phase1.snps.high_confidence.hg38.vcf.gz   -O recal_data.table
```

```bash
gatk ApplyBQSR   -R Homo_sapiens_assembly38.fasta   -I sample.sorted.markdup.bam   --bqsr-recal-file recal_data.table   -O sample.recalibrated.bam
```

BQSR corrects systematic sequencing errors and improves variant calling accuracy.

## 7. Contamination estimation with GATK4.4

```bash
gatk GetPileupSummaries   -I normal.recalibrated.bam   -V small_exac_common_3.hg38.vcf.gz   -L small_exac_common_3.hg38.vcf.gz   -O normal.getpileupsummaries.table
```

```bash
gatk GetPileupSummaries   -I tumor.recalibrated.bam   -V small_exac_common_3.hg38.vcf.gz   -L small_exac_common_3.hg38.vcf.gz   -O tumor.getpileupsummaries.table
```

```bash
gatk CalculateContamination   -I tumor.getpileupsummaries.table   -matched normal.getpileupsummaries.table   -tumor-segmentation segments.table   -O calculatecontamination.table
```

This step estimates sample contamination levels prior to somatic variant calling.

## 8. Somatic variant calling with Mutect2

```bash
gatk Mutect2   -R Homo_sapiens_assembly38.fasta   -I tumor.recalibrated.bam -tumor tumor_sample   -I normal.recalibrated.bam -normal normal_sample   -pon 1000g_pon.hg38.vcf.gz   --germline-resource af-only-gnomad.hg38.vcf.gz   -L wgs_calling_regions.hg38.interval_list   -O somatic.vcf.gz   -bamout tumor_normal.bam
```

Mutect2 identifies candidate somatic variants from matched tumor–normal samples.

## 9. Variant filtering with GATK 4.4

```bash
gatk FilterMutectCalls   -R Homo_sapiens_assembly38.fasta   -V somatic.vcf.gz   --contamination-table calculatecontamination.table   --stats somatic.vcf.gz.stats   --tumor-segmentation segments.table   -O somatic_oncefiltered.vcf.gz
```

This step removes likely false-positive somatic variant calls.

## 10. Retaining high-confidence variants with GATK 4.4

```bash
gatk SelectVariants   -R Homo_sapiens_assembly38.fasta   -V somatic_oncefiltered.vcf.gz   -select "vc.isNotFiltered()"   -O somatic_oncefiltered_SelectVariants-filtered.vcf
```

Only high-confidence (PASS) variants are retained for downstream analysis.

## 11. Variant annotation with Annovar

```bash
time table_annovar.pl   somatic_oncefiltered_SelectVariants-filtered.vcf   humandb/   -buildver hg38   -out sample_file   -remove   -protocol refGene,cytoBand,exac03,avsnp147,dbnsfp30a   -operation gx,r,f,f,f   -nastring .   -vcfinput   -polish   && echo "** annotation done **"
```

Variants are functionally annotated to identify protein-coding mutations.

## 12. Gene expression quantification with Kallisto

```bash
kallisto index -i kallisto_index Homo_sapiens.GRCh38.cdna.all.fa
```

```bash
kallisto quant -i kallisto_index -o expression_output RNAseq_R1_cutadapted.fastq.gz RNAseq_R2_cutadapted.fastq.gz
```

Gene expression levels are quantified for downstream neoantigen filtering.

## 13. HLA typing with OptiType

```bash
razers3 -i 95 -m 1 -dr 0 -o fished_1.bam hla_reference_dna.fasta sample_paired.fastq
samtools bam2fq fished_1.bam > sample_paired_fished.fastq
python OptiTypePipeline.py -i sample_fished_1.fastq sample_fished_2.fastq --dna -v -o optitype_output
```

OptiType predicts patient-specific HLA class I alleles.

## 14. Peptide–HLA binding prediction with NetMHCpan-4.1

Peptide–HLA binding affinity is predicted using the NetMHCpan web service:

- https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/

## 15. Immunogenicity prediction using CNNeo

Immunogenicity prediction is performed using the CNNeo submodel notebooks provided in the models/ directory.

## 16. Rank-level consensus integration and final prioritization (CNNeoPP)

CNNeoPP integrates immunogenicity prediction results from CNNeo using a rank-level consensus strategy to generate the final prioritized list of peptide–HLA pairs.

Specifically, this step consists of the following procedures:

1. The top 50 candidates from each submodel are first selected to form model-specific high-confidence candidate sets.
2. Peptides that appear in at least two submodels are identified as consensus peptides.
3. These consensus peptides are prioritized at the top of the final CNNeo ranking, reflecting consistent support across heterogeneous models.
4. The remaining peptides (supported by only one submodel) are subsequently appended in descending order according to their respective model-specific scores.

This rank-level consensus design emphasizes cross-model agreement rather than absolute score thresholds, effectively reducing false positives while preserving true immunogenic neoantigen candidates.