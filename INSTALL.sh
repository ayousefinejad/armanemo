echo """Upgrading pip ..."""
pip config set global.index http://pypi.partdp.ir/root/pypi &&\
pip config set global.index-url http://pypi.partdp.ir/root/pypi/+simple/ &&\
pip config set global.trusted-host pypi.partdp.ir

pip install --upgrade pip


echo """pip has been updated ..."""

cd dist
echo """Emotion Detection installing ..."""
pip install -r requirements.txt
pip install ./dist/optimum-1.4.0.dev0-py3-none-any.whl
echo """Emtoion Detection installed"""

export BROKER_ADDRESS=192.168.33.76
export LOGGER_BROKER_ADDRESS=192.168.33.77
export PYTHONPATH="${PYTHONPATH}:/~/emotion_detection/"

cd ..

echo """Moving data and model to home directory ..."""
mv data ~/
mv assets ~/
echo """data and model moved to home directory ..."""

