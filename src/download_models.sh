curl -O http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
mkdir "./dataset/models/inception5h"
unzip - inception5h.zip -d "./dataset/models/inception5h/"
rm inception5h.zip

curl -O https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
mkdir "./dataset/models/mobilenet_v2_1.0_224"
tar -xzvf mobilenet_v2_1.0_224.tgz -C "./dataset/models/mobilenet_v2_1.0_224/"
rm mobilenet_v2_1.0_224.tgz
