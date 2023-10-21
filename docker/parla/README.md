# Install PARLA on top of a TPS GPU ENV

Example for building this image

```
export BASE_IMAGE=pecosut/tps_gpu_env:sm_80
export TAG=pecosut/tps_gpu_env_parla:sm_80
export PARLA_TOKEN=<personal token to clone the parla experimental repo>
sudo docker build -t $TAB \
                  --build-arg base_image=$BASE_IMAGE \
                  --build-arg parla_token=$PARLA_TOKEN .
```