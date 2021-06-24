# snzs
Реализация SincNet (arxiv.org/pdf/1808.00158.pdf) с быстрыми сверточными sinc-фильтрами
Основная идея заключается в параметрах - автоматически настраиваемых частотах спектрограммы: по сути, нейросеть
сама отдает предпочтения определенному диапазону частот, в отличие от, например, мел-фильтра, который настроен
на человеческое восприятие звука

В статье рассматривается задача идентификации спикера. Авторы отметили быструю сходимость, которая зависит так же
и от малого числа параметров фильтра (в стандартной свертке кол-во параметров = длине свертки, тут же
не зависит от длины свертки и равно 2 на каждый входной сигнал)

В моем случае к цифрам из статьи не удалось приблизится из-за малого количества эпох. Для улучшения же результата
следует облегчить сеть и увеличить на значительное количество время обучения