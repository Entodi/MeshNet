#!/bin/bash
DATADIR=$1
OUTDIR=$2


CURDIR=`pwd`
cd $DATADIR

if [ ! -f aparc+aseg.nii ]; then
   mri_convert aparc+aseg.mgz ${OUTDIR}/aparc+aseg.nii
fi

if [ ! -f T1.nii ]; then
   mri_convert T1.mgz ${OUTDIR}/T1.nii
   3dUnifize -input ${OUTDIR}/T1.nii -prefix ${OUTDIR}/T1_U.nii
fi

cd $OUTDIR

if [ ! -f all_wmN.nii ]; then
   3dcalc -a aparc+aseg.nii -expr 'equals(a,2)+equals(a,41)+equals(a,7)+equals(a,16)+equals(a,46)+and(step(a-250),step(256-a))' -prefix all_wmN.nii
fi

if [ ! -f all_gmN.nii ]; then
   3dcalc -a aparc+aseg.nii -expr 'and(step(a-1000),step(1036-a))+and(step(a-2000),step(2036-a))+and(step(a-7),step(14-a))+and(step(a-16),step(21-a))+and(step(a-25),step(29-a))+and(step(a-46),step(56-a))+and(step(a-57),step(61-a))' -prefix all_gmN.nii
fi

if [ ! -f gm_wm.nii ]; then
	3dcalc -a all_gmN.nii -b all_wmN.nii -expr 'a+2*b' -prefix gm_wm.nii
fi

#workon dante
python << END
from nipy import save_image, load_image
import numpy as np
T1 = load_image('${OUTDIR}/T1.nii')
gm_wm = load_image('${OUTDIR}/gm_wm.nii')


np.save('${OUTDIR}/affine.npy',T1.affine)
np.save('${OUTDIR}/T1.npy', T1.get_data())
np.save('${OUTDIR}/gm_wm.npy', gm_wm.get_data())

END

rm -rf ${OUTDIR}/all_wm.nii
rm -rf ${OUTDIR}/all_gm.nii
rm -rf ${OURDIR}/gm_wm.nii

cd $CURDIR
