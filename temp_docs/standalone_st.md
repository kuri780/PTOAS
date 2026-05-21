Minimum commands to run a single standalone st


```
# 0) env
cd /workdir/ptoas_a5
source set_ptoas_env.sh
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${ASCEND_HOME_PATH}/runtime/lib64/stub:${LD_LIBRARY_PATH}"

# 1) build (tadd currently fails here; tload succeeds)
ST=/workdir/ptoas_a5/test/tilelang_st/npu/a5/src/st
cd "$ST" && rm -rf build && mkdir build && cd build
cmake .. -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 -DTEST_CASE=tadd \
  -DPTOAS_BIN=/workdir/ptoas_a5/build/tools/ptoas/ptoas
make -j"$(nproc)" tadd          # ← ✅  works now with beta1

export LD_LIBRARY_PATH="${ST}/build/lib:${LD_LIBRARY_PATH}"

# 2) gen golden + inputs
WORK="${ST}/build/testcase/tadd"
mkdir -p "$WORK"
cp "${ST}/testcase/st_common.py" "$WORK/"
cp "${ST}/testcase/tadd/"{cases.py,gen_data.py,compare.py} "$WORK/"
cd "$WORK" && python3 gen_data.py   # ✅ verified

# 3) run main (blocked until build succeeds)
../../bin/tadd                      # ✅  runs CA model now

# 4) validate
python3 compare.py                  # ✅ verified
```


Equivalent, plain CLI, no cmake/make:


```bash
# 0) env
cd /workdir/ptoas_a5
source set_ptoas_env.sh
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${ASCEND_HOME_PATH}/runtime/lib64/stub:${LD_LIBRARY_PATH}"

ST=/workdir/ptoas_a5/test/tilelang_st/npu/a5/src/st
TC="$ST/testcase/tadd"
BUILD="$ST/build"
PTOAS=/workdir/ptoas_a5/build/tools/ptoas/ptoas

# 1) build (plain commands — no cmake/make)
rm -rf "$BUILD" && mkdir -p "$BUILD/bin" "$BUILD/lib" "$BUILD/testcase/tadd"
cd "$BUILD/testcase/tadd"

# 1a) ptoas: tadd.pto -> tadd_kernel.o
"$PTOAS" --pto-arch=a5 --pto-backend=vpto --enable-insert-sync --enable-tile-op-expand \
  "$TC/tadd.pto" -o tadd_kernel.o

# 1b) compile launch.cpp + link kernel shared library
bisheng -fPIC -D_FORTIFY_SOURCE=2 -O2 -std=c++17 \
  -Wno-macro-redefined -Wno-ignored-attributes -Wno-unknown-attributes \
  -fstack-protector-strong -fPIC \
  -xcce -Xhost-start -Xhost-end \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  --cce-aicore-arch=dav-c310-vec -std=gnu++17 \
  -Dtadd_kernel_EXPORTS \
  -I"${ASCEND_HOME_PATH}/include" \
  -I/usr/local/Ascend/driver/kernel/inc \
  -I"${ASCEND_HOME_PATH}/pkg_inc" \
  -I"${ASCEND_HOME_PATH}/pkg_inc/profiling" \
  -I"${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
  -c "$TC/launch.cpp" -o launch.cpp.o

bisheng -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link -shared \
  -Wl,-soname,libtadd_kernel.so \
  -o ../../lib/libtadd_kernel.so launch.cpp.o tadd_kernel.o

# 1c) compile main.cpp + link host executable
bisheng -fPIE -D_FORTIFY_SOURCE=2 -O2 -std=c++17 \
  -Wno-macro-redefined -Wno-ignored-attributes -Wno-unknown-attributes \
  -fstack-protector-strong -fPIC \
  -xc++ -include stdint.h -include stddef.h -std=gnu++17 \
  -I"${ASCEND_HOME_PATH}/include" \
  -I/usr/local/Ascend/driver/kernel/inc \
  -I"$ST/common" \
  -c "$TC/main.cpp" -o main.cpp.o

bisheng -s -Wl,-z,relro -Wl,-z,now main.cpp.o -o ../../bin/tadd \
  -L"${ASCEND_HOME_PATH}/lib64" \
  -L"${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib" \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${BUILD}/lib" \
  ../../lib/libtadd_kernel.so \
  -lruntime_camodel -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase -lpthread

export LD_LIBRARY_PATH="${BUILD}/lib:${LD_LIBRARY_PATH}"

# 2) gen golden + inputs
WORK="${BUILD}/testcase/tadd"
mkdir -p "$WORK"
cp "${ST}/testcase/st_common.py" "$WORK/"
cp "${TC}/"{cases.py,gen_data.py,compare.py} "$WORK/"
cd "$WORK" && python3 gen_data.py

# 3) run main
../../bin/tadd

# 4) validate
python3 compare.py
```
