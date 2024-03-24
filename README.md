# Multiscale Quantum Computing
Multiscale Quantum Computing Codes for Airbus BMW group Quantum Computing Challenge 2024 (ABQCC). 

# requirements:

## pyscf  
[https://pyscf.org/](https://pyscf.org/)

```
pip install pyscf
```

## openfermion 
[https://github.com/quantumlib/OpenFermion](https://github.com/quantumlib/OpenFermion)

```
python -m pip install --user openfermion
```

## openbabel
[https://openbabel.org/index.html](https://openbabel.org/index.html)

```
conda install openbabel -c conda-forge 
pip install -U openbabel
```

## dmet library:

### qc-dmet
[https://github.com/SebWouters/QC-DMET.git](https://github.com/SebWouters/QC-DMET.git)
A modified verson which supports python3 and pyscf >= 2.0 is available at [https://github.com/mahuan-git/QC-DMET-py3.git](https://github.com/mahuan-git/QC-DMET-py3.git)
### lib-dmet (optional)
[https://github.com/zhcui/libdmet_preview.git](https://github.com/zhcui/libdmet_preview.git)

## quantum computing quantum chemistry packages:
### VQEChem
[https://github.com/mahuan-git/VQEChem.git](https://github.com/mahuan-git/VQEChem.git)

### q2chemistry
[https://git.ustc.edu.cn/auroraustc/q2chemistry](https://git.ustc.edu.cn/auroraustc/q2chemistry)


# Installation
```
python3 -m pip install -e .
```

# To be developed:

1. optimization of rdm calculation
2. debug : sci solver for dmet
3. q2chemistry solver 
