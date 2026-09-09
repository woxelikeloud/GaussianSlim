source ~/mylibs/emsdk/emsdk_env.sh

cd 3DGS_Render_glb_unpack
bash build_wasm_new.sh

cp output/gaussianslim_wasm.js ../src/loaders/compactLoader/
cp output/gaussianslim_wasm.wasm ../src/loaders/compactLoader/
cp output/gaussianslim_wasm_pthread.js ../src/loaders/compactLoader/
cp output/gaussianslim_wasm_pthread.wasm ../src/loaders/compactLoader/
if [[ -f output/gaussianslim_wasm_pthread.worker.js ]]; then
    cp output/gaussianslim_wasm_pthread.worker.js ../src/loaders/compactLoader/
fi
cp output/gaussianslim_reconstruction_wasm.js ../src/loaders/compactLoader/
cp output/gaussianslim_reconstruction_wasm.wasm ../src/loaders/compactLoader/

cd ..
npm run build
