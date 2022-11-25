import copy
import math
import numpy.linalg as lin


class SMO:

    @staticmethod
    def simple_flow(lamb: float) -> tuple[str, str, str]:
        P0 = math.exp(-lamb)
        P1, P_rest = lamb * P0, 1 - P0

        return f'Вероятность, что ни одного вызова не будет: {round(P0 * 100, 3)}%', \
               f'Вероятность ровно одного вызова: {round(P1 * 100, 3)}%', \
               f'Вероятность одного вызова или больше: {round(P_rest * 100, 3)}%'

    @staticmethod
    def peak_probabilities(l01: int, l02: int, l10: int, l13: int,
                           l20: int, l23: int, l31: int, l32: int, N=1, flag=None) -> list:

        system = [[l01 + l02, -l10 * N, -l20 * N, 0],
                  [-l01, l10 * N + l13, 0, -l31 * N],
                  [-l02, 0, l20 * N + l23, -l32 * N],
                  [1, 1, 1, 1]]

        system_det = lin.det(system)

        def apply_kramer(matrix: list, col: int) -> int:
            copied_matrix = copy.deepcopy(matrix)
            for i in range(len(matrix[0]) - 1):
                copied_matrix[i][col] = 0
            copied_matrix[-1][col] = 1

            return lin.det(copied_matrix)

        if flag is None:
            result = []
            for column in range(len(system)):
                result.append(
                    f'Предельная вероятность p{column} = '
                    f'{round((apply_kramer(system, column) / system_det) * 100, 3)}%')

            return result

        elif flag == 'continue':
            result = []
            for column in range(len(system)):
                result.append(round((apply_kramer(system, column) / system_det), 3))

            return result

    @staticmethod
    def avg_clean_income(l01: int, l02: int, l10: int, l13: int,
                         l20: int, l23: int, l31: int, l32: int,
                         f1: int, f2: int, b1: int, b2: int, N=1) -> tuple[str, str, str]:

        probs = SMO.peak_probabilities(l01, l02, l10, l13, l20, l23, l31, l32, 1, 'continue')
        no_fix_income = (probs[0] + probs[2]) * f1 + (probs[0] + probs[1]) * f2 - \
                        (probs[1] + probs[3]) * b1 - (probs[2] + probs[3]) * b2

        probs_fix = SMO.peak_probabilities(l01, l02, l10, l13, l20, l23, l31, l32, N, 'continue')
        fix_income = (probs_fix[0] + probs_fix[2]) * f1 + (probs_fix[0] + probs_fix[1]) * f2 - \
                     (probs_fix[1] + probs_fix[3]) * b1 - (probs_fix[2] + probs_fix[3]) * b2

        return f'Доход за единицу времени без ускорения ремонта узлов: ' \
               f'{round(no_fix_income, 3)} ден. ед.', \
               f'Доход за единицу времени с ускорением ремонта узлов: ' \
               f'{round(fix_income, 3)} ден. ед.', \
               f'Изменение дохода за единицу времени: ' \
               f'{int(((fix_income - no_fix_income) / no_fix_income) * 100)}%'

    @staticmethod
    def single_channel_SMO(lamb: int, Tob: int) -> tuple[str, str, str, str]:
        mu = (1 / Tob) * 60
        Q = mu / (lamb + mu)
        P_deny, A = 1 - Q, lamb * Q

        return f'Интенсивность потока обслуживаний: {round(mu, 3)} заявок в час', \
               f'Относительная пропусканая способность: {int(Q * 100)}%', \
               f'Вероятность отказа: {int(P_deny * 100)}%', \
               f'Абсолютная пропускная способность: {round(A, 3)} заявок в час'

    @staticmethod
    def multiple_channel_SMO(lamb: int, Tob: int, N: int, n=10,
                             amount=100) -> tuple[str, list[tuple[str, str, str, str, str]]]:

        mu, goal, result = (1 / Tob) * 60, N / amount, []
        ro = lamb / mu
        p0 = 1

        for i in range(1, n):
            p0 += (ro ** i) / math.factorial(i)
            P_deny = ((ro ** i) / math.factorial(i)) * (p0 ** -1)
            Q = 1 - P_deny
            A = lamb * Q
            k = A / mu
            result.append((f'{i} канал',
                           f'Вероятность отказа: {round(P_deny * 100, 3)}%',
                           f'Относительная пропускная способность: {round(Q * 100, 3)}%',
                           f'Абсолютная пропускная способность: {round(A, 3)} заявок',
                           f'Среднее число занятых каналов: {round(k, 3)}'))

            if Q >= goal:
                break

        return f'Оптимальное число каналов: {i}', result

    @staticmethod
    def single_channel_SMO_with_limited_delay(m: int, T_arr: int, Tob: int) \
            -> tuple[str, str, str, str, str, str, str]:
        lamb, mu = 1 / T_arr, 1 / Tob
        ro = lamb / mu

        if ro != 1:
            P_deny = ((ro ** (m + 1)) * (1 - ro)) / (1 - (ro ** (m + 2)))
            Q = 1 - P_deny
            L_och = ((ro ** 2) * (1 - (ro ** m) * (m + 1 - m * ro))) / \
                    ((1 - ro) * (1 - ro ** (m + 2)))
        else:
            P_deny = 1 / (m + 2)
            Q = 1 - P_deny
            L_och = (m * (m + 1)) / (2 * (m + 2))

        A = lamb * Q
        T_och = L_och / lamb

        return f'Интенсивность входящего потока: {round(lamb, 3)} машин в минуту', \
               f'Интенсивность потока обслуживания: {round(mu, 3)} машин в минуту', \
               f'Интенсивность нагрузки канала: {round(ro, 3)}', \
               f'Вероятность отказа: {round(P_deny * 100, 3)}%', \
               f'Относительная пропускная способность: {round(Q * 100, 3)}%', \
               f'Среднее число машин, ожидающих в очереди на заправку: {round(L_och, 3)}', \
               f'Среднее время ожидания машины в очереди: {round(T_och, 3)} минуты'

    @staticmethod
    def single_channel_SMO_with_unlimited_delay(
            Tob: int, T_avg: int, price: int,
            income: float, time_start: int, time_finish: int) \
            -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:

        lamb, mu = price / T_avg, (1 / Tob) * 60
        ro = lamb / mu
        p0 = 1 - ro
        p_busy, L_och = 1 - p0, (ro ** 2) / (1 - ro)
        T_och = L_och / lamb
        T_SMO = T_och + Tob
        each_time = p_busy * (time_finish - time_start) * 60
        clients_served = each_time / Tob
        daily_income = clients_served * price
        clean_income = daily_income * income

        return f'Интенсивность входящего потока: {round(lamb, 3)} клиента в час', \
               f'Интенсивность потока обслуживания: {round(mu, 3)} клиента в час', \
               f'Интенсивность нагрузки канала: {round(ro, 3)}', \
               f'Вероятность простоя: {round(p0 * 100, 3)}%', \
               f'Вероятность того, что канал занят: {round(p_busy * 100, 3)}%', \
               f'Среднее число клиетов в очереди: {round(L_och, 3)}', \
               f'Среднее время ожидания в очереди: {round(T_och, 3)}', \
               f'Среднее время пребывания: {round(T_SMO, 3)} минут', \
               f'Суммарное время обслуживания: {round(each_time, 3)} минут', \
               f'Количество обслуживаний: {round(clients_served, 3)}', \
               f'Ежедневная выручка: {round(daily_income, 3)} руб.', \
               f'Ежедневный чистый доход: {round(clean_income, 3)} руб.'


def main():
    task = int(input())
    match task:
        case 1:
            print(*SMO.simple_flow(float(input())), sep='\n')
        case 2:
            print(*SMO.peak_probabilities(int(input()), int(input()), int(input()), int(input()),
                                          int(input()), int(input()), int(input()), int(input())), sep='\n')
        case 3:
            print(*SMO.avg_clean_income(int(input()), int(input()), int(input()), int(input()), int(input()),
                                        int(input()),
                                        int(input()), int(input()), int(input()), int(input()), int(input()),
                                        int(input()),
                                        int(input())), sep='\n')
        case 4:
            print(*SMO.single_channel_SMO(int(input()), int(input())), sep='\n')
        case 5:
            result = SMO.multiple_channel_SMO(int(input()), int(input()), int(input()))
            print(result[0])
            print()
            for res in result[1]:
                for line in res:
                    print(line, end='; ')
                print()
        case 6:
            print(*SMO.single_channel_SMO_with_limited_delay(int(input()), int(input()), int(input())), sep='\n')
        case 7:
            print(*SMO.single_channel_SMO_with_unlimited_delay(int(input()), int(input()), int(input()), float(input()),
                                                               int(input()), int(input())), sep='\n')


if __name__ == '__main__':
    main()
