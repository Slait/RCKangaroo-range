(c) 2024, RetiredCoder (RC), Slait

# RCKangaroo v3.2

**RCKangaroo** — это высокопроизводительная реализация метода Кенгуру (Pollard's Kangaroo) для решения задачи дискретного логарифмирования на эллиптических кривых (ECDLP) с использованием графических процессоров NVIDIA (CUDA).

Программа оптимизирована для современных видеокарт и поддерживает CUDA 13.0.

**Авторы:** RetiredCoder, Slait  
**Репозиторий:** https://github.com/Slait/RCKangaroo-range

## Особенности
- **Высокая скорость**: Использует мощь GPU для параллельных вычислений.
- **Поддержка современных архитектур**: Оптимизировано для NVIDIA Turing (RTX 20xx), Ampere (RTX 30xx), Ada Lovelace (RTX 40xx) и Hopper (H100).
- **Гибкая настройка**: Поддержка поиска в заданном диапазоне (Start/End) или по битовой длине.
- **Метод Distinguished Points (DP)**: Эффективное управление памятью и обнаружение коллизий.
- **Кроссплатформенность**: Поддержка Windows и Linux.

## Системные требования
- **ОС**: Windows 10/11 (64-bit) или Linux.
- **GPU**: Видеокарта NVIDIA с поддержкой Compute Capability 7.5 или выше (RTX 20xx и новее).
- **Драйверы**: Установленные драйверы NVIDIA с поддержкой CUDA 13.0.

## Использование

Запуск производится из командной строки `cmd` или `PowerShell`.

### Основные аргументы

*   `-gpu <список>` — Указывает индексы GPU, которые будут использоваться.
    *   Пример: `-gpu 0` (только первая карта), `-gpu 0,1,2` (первые три карты).
*   `-dp <число>` — Задает количество бит для Distinguished Points (14-60).
    *   Влияет на частоту сохранения точек в память. Чем больше значение, тем реже сохраняются точки и тем меньше требуется оперативной памяти (RAM), но увеличивается время на обнаружение коллизии. Рекомендуемое значение: 16-20.
*   `-range <биты>` — Задает диапазон поиска как степень двойки ($2^N$). Используется для бенчмарка или поиска в диапазоне от 0 до $2^N$.
*   `-start <hex>` — Начальное значение диапазона поиска (в шестнадцатеричном формате).
*   `-end <hex>` — Конечное значение диапазона поиска (в шестнадцатеричном формате).
    *   Если указаны `-start` и `-end`, программа автоматически вычислит необходимый `-range`.
*   `-pubkey <hex>` — Публичный ключ, для которого нужно найти приватный ключ (в шестнадцатеричном формате, сжатый или несжатый).
*   `-tames <файл>` — Имя файла для сохранения/загрузки "ручных" (tame) кенгуру (для режима генерации).
*   `-max <число>` — Максимальное количество операций (в единицах $2^{range}$).

### Примеры запуска

#### 1. Режим Benchmark (тест скорости)
Запуск теста скорости на видеокарте 0 в диапазоне 78 бит:
```bash
RCKangaroo.exe -gpu 0 -dp 16 -range 78
```

#### 2. Поиск ключа в заданном диапазоне (Ваш пример)
Поиск приватного ключа для публичного ключа `0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483` в диапазоне от `200000000000000000` до `400000000000000000` (HEX).

```bash
RCKangaroo.exe -gpu 0 -dp 16 -start 200000000000000000 -end 400000000000000000 -pubkey 0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483
```

**Пояснение к примеру:**
*   `-gpu 0`: Используем первую видеокарту.
*   `-dp 16`: Оптимальное значение DP для баланса памяти/скорости.
*   `-start ...`: Начало диапазона (HEX).
*   `-end ...`: Конец диапазона (HEX). Программа автоматически вычислит размер диапазона.
*   `-pubkey ...`: Целевой публичный ключ (сжатый формат).

Sample command line for puzzle #85:

RCKangaroo.exe -dp 16 -range 84 -start 200000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a

Sample command to generate tames:

RCKangaroo.exe -dp 16 -range 76 -tames tames76.dat -max 10

Then you can restart software with same parameters to see less K in benchmark mode or add "-tames tames76.dat" to solve some public key in 76-bit range faster.

<b>Some notes:</b>

Fastest ECDLP solvers will always use SOTA/SOTA+ method, as it's 1.4/1.5 times faster and requires less memory for DPs compared to the best 3-way kangaroos with K=1.6. 
Even if you already have a faster implementation of kangaroo jumps, incorporating SOTA method will improve it further. 
While adding the necessary loop-handling code will cause you to lose about 5–15% of your current speed, the SOTA method itself will provide a 40% performance increase. 
Overall, this translates to roughly a 25% net improvement, which should not be ignored if your goal is to build a truly fast solver. 


<b>Changelog:</b>

v3.2:

- Add Cuda 13.0
- Add start and end range
- Add Russian Translate

v3.1:

- fixed "gpu illegal memory access" bug.
- some small improvements.

v3.0:

- added "-tames" and "-max" options.
- fixed some bugs.

v2.0:

- added support for 30xx, 20xx and 1xxx cards.
- some minor changes.

v1.1:

- added ability to start software on 30xx cards.

v1.0:

- initial release.