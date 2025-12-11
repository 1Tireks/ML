import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # Сортируем по признаку
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]
    
    # Находим уникальные значения и пороги (среднее между соседними)
    unique_features = np.unique(sorted_features)
    
    # Если все значения одинаковые, возвращаем пустые результаты
    if len(unique_features) <= 1:
        return np.array([]), np.array([]), None, None
    
    # Пороги - среднее между соседними уникальными значениями
    thresholds = (unique_features[:-1] + unique_features[1:]) / 2.0
    
    # Векторизованное вычисление критерия Джини для всех порогов
    n = len(feature_vector)
    n_thresholds = len(thresholds)
    
    # Для каждого порога вычисляем количество элементов слева и справа
    # Используем broadcasting для сравнения всех элементов со всеми порогами
    comparisons = sorted_features[:, np.newaxis] < thresholds[np.newaxis, :]  # shape: (n, n_thresholds)
    
    # Количество элементов слева для каждого порога
    n_left = np.sum(comparisons, axis=0)  # shape: (n_thresholds,)
    n_right = n - n_left  # shape: (n_thresholds,)
    
    # Фильтруем пороги, где одно из поддеревьев пустое
    valid_mask = (n_left > 0) & (n_right > 0)
    if not np.any(valid_mask):
        return np.array([]), np.array([]), None, None
    
    thresholds = thresholds[valid_mask]
    n_left = n_left[valid_mask]
    n_right = n_right[valid_mask]
    comparisons = comparisons[:, valid_mask]
    
    # Вычисляем долю класса 1 в левом поддереве для каждого порога
    # sorted_targets == 1 дает булеву маску, умножаем на comparisons для получения левых элементов
    left_class1 = np.sum((sorted_targets == 1)[:, np.newaxis] * comparisons, axis=0)  # shape: (n_thresholds,)
    p1_left = left_class1 / n_left
    p0_left = 1 - p1_left
    H_left = 1 - p1_left**2 - p0_left**2
    
    # Вычисляем долю класса 1 в правом поддереве для каждого порога
    right_class1 = np.sum((sorted_targets == 1)[:, np.newaxis] * (~comparisons), axis=0)  # shape: (n_thresholds,)
    p1_right = right_class1 / n_right
    p0_right = 1 - p1_right
    H_right = 1 - p1_right**2 - p0_right**2
    
    # Критерий Джини (отрицательный, так как мы максимизируем)
    ginis = -(n_left / n) * H_left - (n_right / n) * H_right
    
    # Находим лучший порог (максимальный критерий Джини, при равенстве - минимальный порог)
    best_idx = np.argmax(ginis)
    # Если есть несколько с одинаковым максимумом, выбираем минимальный
    max_gini = ginis[best_idx]
    best_indices = np.where(ginis == max_gini)[0]
    best_idx = best_indices[0]  # Выбираем первый (минимальный порог)
    
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        # Критерий останова: все объекты одного класса
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        # Критерий останова: достигнута максимальная глубина
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        # Критерий останова: недостаточно образцов для разбиения
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    # Исправление: деление должно быть current_click / current_count
                    if current_count > 0:
                        ratio[key] = current_click / current_count
                    else:
                        ratio[key] = 0
                sorted_categories = sorted(ratio.items(), key=lambda x: x[1])
                sorted_categories = [x[0] for x in sorted_categories]
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            # Пропускаем константные признаки
            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None or gini is None:
                continue
                
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [x[0] for x in categories_map.items() if x[1] < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Проверка min_samples_leaf
        left_mask = split
        right_mask = ~split
        if (self._min_samples_leaf is not None and 
            (np.sum(left_mask) < self._min_samples_leaf or np.sum(right_mask) < self._min_samples_leaf)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"], depth + 1)
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]
        
        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
