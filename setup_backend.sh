# tested in the past with ubuntu 22.04 and python 3.9.16 but not for a long time now
# most current windows enviroment runs Python 3.10.1 for which the following packages have to be modified
# or test it with 3.9.16, it *should* work, I don't think there's anything 3.10 specific in here
sudo apt update

if [ ! -z $1 ] && [ "$1" = "--install-backend-audio-unstable-3.9" ]
then
    sudo apt-get install -y build-essential python3.9-dev libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0
    # sudo apt-get install libav-tools
    pip install simpleaudio
    pip install pyaudio
fi

pip install numpy
pip install scipy
pip install soundfile

pip install flask
pip install flask_cors
pip install eventlet
pip install flask_socketio
pip install deepl
pip install openai
pip install google-cloud-texttospeech

pip install thefuzz
pip install python-Levenshtein