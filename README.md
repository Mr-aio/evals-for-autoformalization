# evals-for-autoformalization

## Get Compute for your Research Project

### Snap Cluster Important References
Always use the original documentation or wiki for each cluster: https://ilwiki.stanford.edu/doku.php?id=start (your **snap bible**).
Other useful resources:
- compute instructions from Professor Koyejo's (Sanmi's) lab (STAIR Lab): https://docs.google.com/document/d/1PSTLJdtG3AymDGKPO-bHtzSnDyPmJPpJWXLmnJKzdfU/edit?usp=sharing
- advanced video from Rylan and Brando (made for the STAIR/Koyejo Lab): https://www.youtube.com/watch?v=XEB79C1yfgE&feature=youtu.be
- support IT for snap: il-action@cs.stanford.edu
- our CS 197 section channel
- join the snap slack & ask questions there too: https://join.slack.com/t/snap-group/shared_invite/zt-1lokufgys-g6NOiK3gQi84NjIK_2dUMQ

### Set up Project in the Snap Cluster - assuming a single Sever
Goal: Set up Project in the Snap Cluster - **assuming a single sever** (as if you were using a single ocmputer, since that is most simple + good enough for essential goal (i.e., doing research).
The details of how to use different storage systems are complicated and you don't need that version right now.
Note snap currently has no slurm (HPC workload manager).

High level plan
1. Vscode ssh into your server, which enables modifying the files on the server directly (other options are usually modify locally and push on save)
2. ssh into your server of choice ```ssh brando9@mercury1.stanford.edu``` (algign_4_af)  or ```ssh brando9@mercury2.stanford.edu``` (equiv prover bench) or ```ssh brando9@hyperturning1.stanford.edu``` (div team)
3. ---
4. create a public ssh key the snap server of choice and then git clone the repo
5. then you need to set up a python env, in this case `conda` and install the projects using `pip install -e .` (and have a rough idea how python packing works)
  i. if `conda` is not available install it here locally in the server you're suing follwing these instructions: https://github.com/brando90/ultimate-utils/blob/master/sh_files_repo/download_and_install_conda.sh (bonus, `module avail` might have it, but it might also be a good thing to ask them to install it for you or why isn't it available)
  ii. create a conda env for your project with a good yet short name (`conda create -n align_4_af python=3.10`)
  iii. put `conda activate` in your `.bashrc.user` file in snap as instructed here https://ilwiki.stanford.edu/doku.php?id=hints:enviroment (so you don't have to run conda activate your_env every time) [TODO: ask it for help or help fix]
6. now let's instlal the library `pip install -e .` or `pip install -e $HOME/evals-for-autoformalization/setup.py`
7. test gpu works by running pytorch (or cpu locally)
8. test some code in your server + nvidia-smi + set visible devices + cuda stuff set up properly
9. ---
10. then understand the workflow for long running jobs: krbtmux, reauth, tmux attach -t 0, tmux ls
11. understand how to modify your code, test the code, and learn to git push to your team's github repo/fork
12. then run a real experiment then repeat

Need to know let's decide later where to put this in the intructions:
- .bashrc + .bashrc details of snap: https://github.com/brando90/.dotfiles 

Bonus:
- kinit for avoiding passwords
- request an optional pull rquest to the original repo
- ampere arch fp32 vs fp 16 and bf16. The goods for ML are bf16 and fp32.


note: you should understand (roughly) what everything means in here to be effective. Google, gpt4/claude it etc. 
Tips:
- use `man` to understand bash command or if you want to chat with it use LLMs/GPT4/Claude and `--help` or `-h`.

List of thinks to know about:
- git, ssh, bash,
- basic unix commands, ls, ls -lah, cd, pwd, which,
- vim
- nvidia-smi
- module load (common in HPC's)

#### 1 Login to Login/head node
```bash
ssh your_sunetid@login.sherlock.stanford.edu
```
get the git clone command from your fork (create your fork in github! To fork go to github our project's webpage and click fork on the top right) and do with your own specific ssh url clone text (gotten from the topish left green button called "code", copy the ssh flag):
```bash
git clone git@github.com:brando90/evals-for-autoformalization.git
```

## SSH
Goal: add the public key you created on sherlock's login node to your github so you can clone your fork. For that follow the instructions here https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account or the outline from bellow that was inspired from the official github link in this sentence.

First create ssh public key on sherlock
```bash
ssh your_sunetid@login.sherlock.stanford.edu
[brando9@sh03-ln06 login ~/.ssh]$ ssh-keygen -t ed25519 -C "brandojazz@gmail.com"
Generating public/private ed25519 key pair.
Enter file in which to save the key (/home/users/brando9/.ssh/id_ed25519):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/users/brando9/.ssh/id_ed25519.
Your public key has been saved in /home/users/brando9/.ssh/id_ed25519.pub.
The key fingerprint is:
...
The key's randomart image is:
+--[ED25519 256]--+
...
+----[SHA256]-----+
# press the enter key to not change file name
# press the enter key or a passphase to use this key
```
Now run ssh agent in sherlock
```
[brando9@sh03-ln06 login ~/.ssh]$ eval "$(ssh-agent -s)"
Agent pid 50895
```
Now configure your .ssh if you've never done it on this server.
Concretely, if ~.ssh/config doesn't exist create it with (or vim): 
```
touch ~/.ssh/config
# or
[brando9@sh03-ln06 login ~/.ssh]$ vim .config
```
put the contets of for hithub (i.e., copy the bellow into your clip board, read it) with the vim:
```
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```
i.e. use vim editor in sherlock(read about vim, it's just an edit) in the server i.e.
do
```
[brando9@sh03-ln06 login ~/.ssh]$ cat ~/.ssh/config
cat: /home/users/brando9/.ssh/config: No such file or directory
vim ~/.ssh/config
# press i in the new black window,
#copy paste the contents above after pressing i,
#then press escape esc
# then safe the file and exist with :x or :w followed by :q
# then do 
cat .config
# confirms you copied it correctly
[brando9@sh03-ln06 login ~/.ssh]$ cat .config
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```
Then add the key to your github using https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account . For a summary of what I did do:
```
# in the sherlock login/head node do:
[brando9@sh03-ln06 login ~/.ssh]$ cat ~/.ssh/id_ed25519.pub
# then copy paste the output, very carefully, do not share this stuff publicly wide web
```
Then go to setting in your github e.g., https://github.com/settings/keys and create a new key by copy pasting the contents of the previous cat command.

Then git clone on your fork should work, e.g.,:
```
[brando9@sh03-ln06 login ~/.ssh]$ git clone git@github.com:brando90/evals-for-autoformalization.git
Cloning into 'evals-for-autoformalization'...
remote: Enumerating objects: 270, done.
remote: Counting objects: 100% (264/264), done.
remote: Compressing objects: 100% (163/163), done.
remote: Total 270 (delta 150), reused 175 (delta 90), pack-reused 6
Receiving objects: 100% (270/270), 78.74 KiB | 0 bytes/s, done.
Resolving deltas: 100% (151/151), done.
```

## Tutorial on setting up a python project
1. create the `setup.py` file
2. Make sure your setup.py file has the following
```python
    package_dir={'': 'src'},
    packages=find_packages('src'),
```
so that `pip install -e .` knows were the python modules are when installing the python library. 
Anything outside `src`` won't be found for this libraries pip -e install.
Read the comments for those lines in `setup.py`` to understand it if you wish and refs.
3. Now you can do `pip install -e .` or `pip install -e $HOME/evals-for-autoformalization` (assuming you have your python env/conda env activated).

Now you should be able to import statements for this library in the expected way! e.g.,
```python
# TODO
```

## Python Envs with conda & using pip instlal -e <path>

```bash
# This script demonstrates the use of conda for managing Python environments and pip for installing Python packages.
# Conda is an open-source package management and environment management system.

# 1. List all the available conda environments on your system.
# The command 'conda info -e' will list all the environments available.
conda info -e

# 2. Update conda to the latest version.
# It's good practice to keep conda updated to the latest version to avoid any compatibility issues.
conda update --all

# 3. Upgrade pip to the latest version.
# Pip is the package installer for Python. Upgrading it ensures that you can install packages without issues.
pip install --upgrade pip

# 4. Create a new conda environment.
# 'conda create -n maf python=3.10' creates a new environment named 'maf' with Python version 3.10 installed.
# '-n maf' specifies the name of the environment, and 'python=3.10' specifies the Python version.
conda create -n af_evals python=3.10

# 5. Activate the newly created conda environment.
# 'conda activate maf' activates the 'maf' environment. Once activated, any Python packages installed will be specific to this environment.
conda activate af_evals

# To deactivate the current environment and return to the base environment, you can use:
# conda deactivate

# If you want to remove the 'maf' environment completely, you can use:
# conda remove --name af_evals --all

# 6. Install Python packages using pip in editable mode.
# 'pip install -e <path>' installs a package in 'editable' mode, meaning changes to the source files will immediately affect the installed package without needing a reinstall.
# Replace '<path>' with the path to the directory containing the 'setup.py' file of the package you want to install.
# pip install -e $HOME/evals-for-autoformalization
pip install -e .

# Test pytorch install with GPU
python -c "import torch; print(torch.version.cuda); print((torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()))"

# Note: It's crucial to activate the correct conda environment before using pip install to avoid installing packages in the wrong environment.
```
ref: https://chat.openai.com/c/375d5d26-7602-4888-9ef5-9f92359330dc

## Basic Git
TODO
```bash
cd /afs/cs.stanford.edu/u/brando9/
git clone git@github.com:brando90/massive-autoformalization-maf.git
ln -s /afs/cs.stanford.edu/u/brando9/massive-autoformalization-maf $HOME/massive-autoformalization-maf
pip install -e ~/massive-autoformalization-maf
#pip uninstall ~/massive-autoformalization-maf
cd ~/massive-autoformalization-maf
```
## VSCode
TODO
  Next session, we need to walk through how to use the VSCode debugger with snap
## Basic Docker

## Basic Cluster use (Snap)

## Lean4

Recommend watching:

https://www.youtube.com/watch?v=yZo6k48L0VY

https://www.youtube.com/watch?v=_0QZXHoyZlA 
