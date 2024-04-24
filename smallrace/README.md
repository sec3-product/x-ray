# Building 

Run the following from the smallrace root directory

```bash
sudo apt-get install libxml2-dev
mkdir -p build
cd build && cmake  -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
