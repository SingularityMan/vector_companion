:: FLASH ATTENTION + STABLE TORCH


:: Uninstall existing flash-attn
pip uninstall flash-attn

:: Install dependencies
pip install packaging
pip install wheel


:: Install ninja for faster compile
pip uninstall -y ninja && pip install ninja


:: clone flash attention
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention


:: set the number of threads. 10 jobs uses 60gigs of ram at peak
set MAX_JOBS=4


:: start compile
:: python setup.py install


:: build wheel
python setup.py bdist_wheel


:: pip install some-package.whl
cd dist
pip install flash_attn-2.6.3-cp312-cp312-win_amd64.whl


pause