# Coherent Imaging, Source Separation, Remote Sensing, and Industrial Applications (IMA207) - 2023/2024

## Course Overview

This repository contains materials and resources for the course **IMA207: Coherent Imaging, Source Separation, Remote Sensing, and Industrial Applications**, part of the **Image-Data-Signal** curriculum. The course introduces advanced topics in imaging, focusing on coherent imaging, source separation, and their applications in remote sensing, while also connecting theory with industrial practices in image analysis.

### Key Topics:
- Coherent Imaging: Introduction to Synthetic Aperture Radar (SAR) imaging, statistical modeling, and noise handling in coherent imaging.
- Source Separation: Techniques such as Non-negative Matrix Factorization (NMF) applied to hyperspectral imaging.
- Remote Sensing: Techniques for optical and multi-spectral imaging, including pan-sharpening and hyperspectral analysis.
- Industrial Applications: Practical seminars on biometrics, non-destructive testing, autonomous vehicles, and more, featuring guest speakers from industry.

## Prerequisites

Students are expected to have knowledge of:
- Basic image processing techniques
- Fundamentals of statistics and linear algebra

## Course Structure

- Total Hours: 24 hours of lectures and practical sessions, plus 4 to 5 industrial seminars.
- Credits: 2.5 ECTS
- Evaluation: Practical work (50%) and final written exam (50%). Attendance is mandatory for practical sessions and guest seminars.

## Instructor

- Professor Christophe Kervazo

## Installation and Setup

Some exercises and projects require Python and relevant image processing libraries. You can follow the instructions below to set up your environment using `conda`:

1. Anaconda/Miniconda: Download and install Python with Anaconda or Miniconda from [Conda Official Site](https://docs.conda.io/en/latest/).
2. Image Processing Libraries: Create a new conda environment with the necessary packages:
   ```bash
   conda create -n ima python matplotlib numpy scipy scikit-image ipykernel pandas scikit-learn jupyter tqdm bokeh opencv munkres
   ```
3. Activate the environment:
   ```bash
   conda activate ima
   ```

4. Launch Jupyter Notebook (if required for exercises):
   ```bash
   jupyter notebook
   ```

This will set up the necessary environment for running image processing tasks and exercises for the course.

## How to Contribute

Feel free to contribute to the repository by:
- Submitting pull requests for corrections or improvements.
- Providing additional examples or extending the projects.
