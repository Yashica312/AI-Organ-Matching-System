#!/usr/bin/env bash

cd frontend
npm install
npm run build

cd ..

rm -rf static/*
mkdir -p static

cp -r frontend/dist/* static/ 2>/dev/null || cp -r frontend/build/* static/
