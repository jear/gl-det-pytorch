echo "export http_proxy=http://172.28.6.17:3128" >> ~/.bashrc
echo "export https_proxy=http://172.28.6.17:3128" >> ~/.bashrc
echo "export ftp_proxy=http://172.28.6.17:3128" >> ~/.bashrc
echo "export no_proxy=localhost,127.0.0.1,.gl-hpe.local,.gwtest.com" >> ~/.bashrc

source ~/.bashrc

conda config --set proxy_servers.http http://172.28.6.17:3128 
conda config --set proxy_servers.https http://172.28.6.17:3128

# Testing
conda create -n my-condaenv python=3.8
conda init
source ~/.bashrc
conda activate my-condaenv
pip install matplotlib
