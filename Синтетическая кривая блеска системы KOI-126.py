import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Константы
G = 6.67430e-11  # гравитационная постоянная, м^3 кг^-1 с^-2
M_sun = 1.98847e30  # масса Солнца, кг

# Параметры системы KOI-126 (внутренняя двойная B–C)
m1 = 0.241 * M_sun  # масса звезды B
m2 = 0.212 * M_sun  # масса звезды C
R1 = 0.254 * 6.957e8  # радиус звезды B в метрах (примерно 0.254 радиуса Солнца)
R2 = 0.231 * 6.957e8  # радиус звезды C в метрах (примерно 0.231 радиуса Солнца)

P = 1.767 * 24 * 3600  # период в секундах

# Расчёт орбитального радиуса (треть закон Кеплера)
a = (G * (m1 + m2) * (P / (2 * np.pi)) ** 2) ** (1 / 3)

# Начальные условия
r1_0 = np.array([-m2 / (m1 + m2) * a, 0])
v1_0 = np.array([0, -np.sqrt(G * m2**2 / (a * (m1 + m2)))])

r2_0 = np.array([m1 / (m1 + m2) * a, 0])
v2_0 = np.array([0, np.sqrt(G * m1**2 / (a * (m1 + m2)))])

# Уравнения движения
def derivatives(t, y):
    r1 = y[:2]
    v1 = y[2:4]
    r2 = y[4:6]
    v2 = y[6:8]

    r = r2 - r1
    distance = np.linalg.norm(r)

    a1 = G * m2 * r / distance**3
    a2 = -G * m1 * r / distance**3

    return [*v1, *a1, *v2, *a2]

# Вектор начальных условий
y0 = np.concatenate([r1_0, v1_0, r2_0, v2_0])

# Временной интервал интегрирования
T = P * 3
num_points = 2000
t_span = (0, T)
t_eval = np.linspace(*t_span, num_points)

# Решение системы
solution = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

# Координаты
r1 = solution.y[:2]
r2 = solution.y[4:6]

# Синтетическая кривая блеска
brightness = []

for i in range(len(t_eval)):
    # Положение звёзд в данный момент
    x1, y1 = r1[:, i]
    x2, y2 = r2[:, i]

    # Проверяем перекрытие вдоль оси Y (наблюдаем вдоль Y)
    dx = np.abs(x1 - x2)
    overlap = dx < (R1 + R2)

    if overlap:
        # Кто впереди по оси наблюдения (ось Y)?
        if y1 > y2:
            # Звезда 1 впереди — видим только её свет
            total_brightness = 1.0  # условно
        else:
            # Звезда 2 впереди — видим только её свет
            total_brightness = 0.9  # допустим, звезда 2 чуть слабее
    else:
        # Нет перекрытия — суммарная яркость
        total_brightness = 1.9  # сумма яркостей обоих объектов

    brightness.append(total_brightness)

# Визуализация кривой блеска
plt.figure(figsize=(8, 4))
plt.plot(t_eval / 3600 / 24, brightness, color='orange')
plt.title('Синтетическая кривая блеска системы KOI-126')
plt.xlabel('Время [дни]')
plt.ylabel('Относительная яркость')
plt.grid(True)
plt.tight_layout()
plt.show()
