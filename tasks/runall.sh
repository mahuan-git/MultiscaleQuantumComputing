for i in POSCAR_acid  POSCAR_oxime  POSCAR_tria  POSCAR_trib
	do
		sbatch python.sh $i
	done
