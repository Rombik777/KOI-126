import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


G = 6.67430e-11  # гравитационная постоянная, м^3 кг^-1 с^-2
M_sun = 1.98847e30  # масса Солнца, кг

# Параметры системы KOI-126 (внутренняя двойная B–C)
m1 = 0.241 * M_sun  # масса звезды B
m2 = 0.212 * M_sun  # масса звезды C

# Орбитальный период ~1.767 дня => пересчитаем в секунды
P = 1.767 * 24 * 3600  # период, сек

# По третьему закону Кеплера находим радиус орбиты (приближенно, круговая орбита)
a = (G * (m1 + m2) * (P / (2 * np.pi))**2) ** (1 / 3)  # м

# Начальные условия
# Тело 1
r1_0 = np.array([-m2 / (m1 + m2) * a, 0])
v1_0 = np.array([0, -np.sqrt(G * m2**2 / (a * (m1 + m2)))])

# Тело 2
r2_0 = np.array([m1 / (m1 + m2) * a, 0])
v2_0 = np.array([0, np.sqrt(G * m1**2 / (a * (m1 + m2)))])

# Система ОДУ
def derivatives(t, y):
    r1 = y[:2]
    v1 = y[2:4]
    r2 = y[4:6]
    v2 = y[6:8]

    # Вектор расстояния и его модуль
    r = r2 - r1
    distance = np.linalg.norm(r)

    # Ускорения
    a1 = G * m2 * r / distance**3
    a2 = -G * m1 * r / distance**3

    return [*v1, *a1, *v2, *a2]

# Начальный вектор состояния
y0 = np.concatenate([r1_0, v1_0, r2_0, v2_0])

# Временной интервал
T = P * 3  # три орбитальных периода
num_points = 1000
t_span = (0, T)
t_eval = np.linspace(*t_span, num_points)

# Решение задачи
solution = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

# Извлекаем траектории
r1 = solution.y[:2]
r2 = solution.y[4:6]

# Визуализация
plt.figure(figsize=(7, 7))
plt.plot(r1[0], r1[1], label='Тело 1 (звезда B)')
plt.plot(r2[0], r2[1], label='Тело 2 (звезда C)')
plt.title('Орбиты двух тел KOI-126 (внутренняя система B–C)')
plt.xlabel('x [м]')
plt.ylabel('y [м]')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.savefig('KOI126_orbits.png', dpi=300)
plt.show()
