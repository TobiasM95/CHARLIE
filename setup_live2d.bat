@echo off

REM Requirements: Node.js (v19.8.1), npm (9.5.1) with npx

pushd .\live2d\Samples\TypeScript\Demo\
npm install
npm run build
popd