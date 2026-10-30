# AVX-512 для применения моделей CatBoost

* [`paper-summary.md`](paper-summary.md) — конспект `paper.pdf` (ВКР
  А. Д. Миронова, 2022): что там измерялось и какие выводы взяты в работу.
* [`implementation.md`](implementation.md) — что реализовано, где лежит, как
  проверять и как замерять.
* [`results.md`](results.md) — замеры. **AVX-512 пока не измерен**: машина
  разработки без AVX-512, в файле только baseline.

Инструменты:

* [`kernel_check/`](kernel_check) — автономная проверка ядер AVX-512; работает и
  без AVX-512 через SIMDe.
* [`benchmark/`](benchmark) — бенчмарк применения модели поверх C API плюс
  скрипт обучения синтетической модели нужной формы.

Быстрый старт:

```sh
nix build                                        # библиотека C API в ./result
./avx512-results/kernel_check/build_and_run.sh   # корректность ядер

nix develop .#benchmark -c python3 avx512-results/benchmark/make_model.py --output /tmp/bench.cbm
./avx512-results/benchmark/build_and_run.sh --model /tmp/bench.cbm --docs 1024 --repetitions 200
```
