import gdown

url = "https://drive.google.com/u/0/uc?id=1NSv4RCSHjcHois3dXjYw_PaLIoVlLgXu&export=download"
output = "colored_mnist.tar.gz"
gdown.download(url, output)

