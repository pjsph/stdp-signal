# stdp-mnist

Carrying <https://github.com/peter-u-diehl/stdp-mnist> from [Brian1](https://github.com/brian-team/brian) to [Brian2](https://github.com/brian-team/brian2).

## Installation

1. Clone the repository on your local machine :
```
git clone https://github.com/pjsph/stdp-mnist.git
```
2. Download MNIST database and put the files in a new **./MNIST/** folder.
3. Make sure you have **./random/**, **./random2/** and **./weights/** folders in your project directory (even if they are empty)
4. Create a new python virtual environment :
```
python -m venv env
```
or see <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>

5. Enter the newly created environment :
```
./env/Scripts/activate
```
6. Install all dependencies :
```
python -m pip install -r requirements.txt
```
7. Start the app :
```
python main.py
```
8. To quit the virtual environment :
```
deactivate
```
