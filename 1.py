import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


def check_file_existence(file_path):
    """
    检查文件是否存在且可读
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 '{file_path}' 不存在。")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"没有权限读取文件 '{file_path}'。")


def load_and_preprocess_data(u_data_path, u_item_path):
    """
    加载并预处理评分数据和电影信息数据
    """
    check_file_existence(u_data_path)
    check_file_existence(u_item_path)

    # 加载评分数据
    ratings = pd.read_csv(u_data_path, sep="\t", names=["UserID", "MovieID", "Rating", "Timestamp"])
    # 加载电影信息
    movies = pd.read_csv(u_item_path, sep="|", encoding="latin - 1", header=None,
                          names=["MovieID", "Title"] + [f"Col_{i}" for i in range(22)])
    # 合并评分数据和电影名称
    ratings = pd.merge(ratings, movies[["MovieID", "Title"]], on="MovieID")
    return ratings


def find_common_movies(ratings_df, target_count):
    """
    统计每个用户评分的电影数量，并取前target_count个用户，
    再统计这些用户评分的电影数量，取前target_count部电影，
    返回筛选出的用户和电影的列表
    """
    if target_count > len(ratings_df['UserID'].unique()) or target_count > len(ratings_df['MovieID'].unique()):
        raise ValueError("target_count 超过了数据集中用户或电影的总数。")
    # 统计每个用户评分的电影数量，并取前target_count个用户
    top_users = ratings_df.groupby('UserID')['MovieID'].nunique().nlargest(target_count).index
    # 统计这些用户评分的电影数量，并取前target_count部电影
    top_movies = ratings_df[ratings_df['UserID'].isin(top_users)].groupby('MovieID')['UserID'].nunique().nlargest(
        target_count).index
    return sorted(top_users.tolist()), sorted(top_movies.tolist())


def calculate_ratings_matrix(ratings_df, selected_users, selected_movies):
    """
    根据传入的用户和电影列表构建评分矩阵，
    如果用户和电影列表不为空，则筛选数据构建并返回评分矩阵，否则返回None
    """
    if not selected_users or not selected_movies:
        return None
    ratings_matrix = ratings_df[(ratings_df['UserID'].isin(selected_users)) & (
            ratings_df['MovieID'].isin(selected_movies))]
    # 将数据转换为pivot表，形成相应的评分矩阵
    ratings_matrix = ratings_matrix.pivot_table(index='UserID', columns='MovieID', values='Rating')
    return ratings_matrix


def find_test_sets(selected_users, selected_movies):
    """
    生成用于划分训练集和测试集的测试集索引，
    通过随机选择一定比例的数据点作为测试集，并返回其索引集合
    """
    all_indices = np.array(np.meshgrid(np.arange(len(selected_users)), np.arange(len(selected_movies)))).T.reshape(
        -1, 2)
    test_size = int(0.3 * len(all_indices))
    np.random.seed(42)  # 设置随机种子
    test_indices = np.random.choice(len(all_indices), test_size, replace=False)
    # 记录测试集索引
    test_set = set(tuple(index) for index in all_indices[test_indices])
    return test_set


def generate_new_ratings_matrix(ratings_matrix, test_set):
    """
    根据测试集索引，从原始评分矩阵中移除对应位置的数据，生成新的评分矩阵，
    用于后续的训练和预测操作
    """
    new_ratings_matrix = ratings_matrix.copy()
    for (i, j) in test_set:
        new_ratings_matrix.iloc[i, j] = np.nan
    return new_ratings_matrix


def fill_missing_values(ratings_matrix):
    """
    填充评分矩阵中的缺失值，计算每个用户的平均评分，
    并用该平均评分替换对应用户的缺失值
    """
    user_mean_ratings = ratings_matrix.mean(axis=1, skipna=True)
    for i in range(ratings_matrix.shape[0]):
        for j in range(ratings_matrix.shape[1]):
            if pd.isnull(ratings_matrix.iloc[i, j]):
                ratings_matrix.iloc[i, j] = np.round(user_mean_ratings.iloc[i], 2)
    return ratings_matrix


def svd_predict(ratings_matrix, k=2):
    """
    对评分矩阵进行SVD分解，并基于分解结果预测评分矩阵，
    返回预测后的评分矩阵
    """
    try:
        U, sigma, Vt = svds(ratings_matrix.values, k=k)
        sigma = np.diag(sigma)
        predictions_matrix = np.dot(np.dot(U, sigma), Vt)
        return predictions_matrix
    except Exception as e:
        raise RuntimeError(f"SVD分解失败: {e}")


def calculate_error(iter_matrix, groundtruth_matrix, test_set, k, max_iterations=1000):
    """
    计算不同迭代次数下预测评分的平均绝对误差（MAE）和均方根误差（RMSE），
    返回MAE和RMSE的历史记录列表
    """
    mae_history = []
    rmse_history = []
    iter_matrix = iter_matrix.copy()
    for _ in range(max_iterations):
        try:
            predictions_matrix = svd_predict(iter_matrix, k=k)
            predicted_ratings = []
            actual_ratings = []
            for (x, y) in test_set:
                predicted_rating = predictions_matrix[x, y]
                iter_matrix.iloc[x, y] = predicted_rating
                actual_rating = groundtruth_matrix.iloc[x, y]
                predicted_ratings.append(predicted_rating)
                actual_ratings.append(actual_rating)

            iteration_mae = mean_absolute_error(actual_ratings, predicted_ratings)
            iteration_mse = mean_squared_error(actual_ratings, predicted_ratings)
            mae_history.append(iteration_mae)
            rmse_history.append(np.sqrt(iteration_mse))
        except Exception as e:
            print(f"计算误差时出错: {e}")
    return mae_history, rmse_history


def recommend_movies_for_user(user_id, predictions_matrix, ratings_df, top_n=5):
    """
    为指定用户推荐电影
    :param user_id: 用户ID
    :param predictions_matrix: 预测评分矩阵
    :param ratings_df: 原始评分数据
    :param top_n: 推荐电影的数量
    :return: 推荐的电影列表（包含电影标题和预测评分）
    """
    # 获取用户的预测评分行
    user_predictions = predictions_matrix[user_id - 1]
    # 获取用户已评分的电影ID
    user_rated_movies = ratings_df[ratings_df['UserID'] == user_id]['MovieID'].tolist()
    # 构建一个字典，键为电影ID，值为预测评分
    movie_prediction_dict = {movie_id: pred for movie_id, pred in zip(ratings_df['MovieID'].unique(), user_predictions)}
    # 过滤掉用户已评分的电影
    unrated_movies = {movie_id: pred for movie_id, pred in movie_prediction_dict.items() if movie_id not in user_rated_movies}
    # 按预测评分降序排序
    sorted_movies = sorted(unrated_movies.items(), key=lambda x: x[1], reverse=True)
    # 取前top_n部电影
    top_movies = sorted_movies[:top_n]
    # 获取电影标题
    movies_df = ratings_df[['MovieID', 'Title']].drop_duplicates()
    recommended_movies = []
    for movie_id, pred in top_movies:
        movie_title = movies_df[movies_df['MovieID'] == movie_id]['Title'].iloc[0]
        recommended_movies.append((movie_title, pred))
    return recommended_movies


def load_data():
    global ratings_df
    file_path = entry_file_path.get()  # 获取用户在界面输入的文件路径
    try:
        check_file_existence(file_path)
        ratings_df = pd.read_csv(file_path)
        text_result.insert(tk.END, "数据加载成功，以下是数据前1000行内容：\n")
        text_result.insert(tk.END, str(ratings_df.head(1000)))  # 只展示前1000行数据
    except (FileNotFoundError, PermissionError) as e:
        messagebox.showerror("错误", f"文件加载出错: {e}")


def process_data():
    global selected_users, selected_movies, ratings_matrix
    target_count = int(entry_target_count.get())  # 获取用户输入的筛选数量
    try:
        selected_users, selected_movies = find_common_movies(ratings_df, target_count)
        text_result.insert(tk.END, f"筛选出的用户数量: {len(selected_users)}, 筛选出的电影数量: {len(selected_movies)}\n")
        ratings_matrix = calculate_ratings_matrix(ratings_df, selected_users, selected_movies)
        if ratings_matrix is None:
            text_result.insert(tk.END, "没有足够的用户和电影数据来构建评分矩阵。")
            return
        text_result.insert(tk.END, "构建的评分矩阵基本信息：\n")
        text_result.insert(tk.END, str(ratings_matrix.info()))
        text_result.insert(tk.END, "构建的评分矩阵前几行：\n")
        text_result.insert(tk.END, str(ratings_matrix.head()))
    except ValueError as e:
        messagebox.showerror("错误", f"数据处理出错: {e}")


def train_and_predict():
    global test_set, predictions_matrix
    try:
        test_set = find_test_sets(selected_users, selected_movies)
        new_ratings_matrix = generate_new_ratings_matrix(ratings_matrix, test_set)
        filled_ratings_matrix = fill_missing_values(new_ratings_matrix)
        k = int(entry_k.get())  # 获取用户输入的SVD分解的k值
        predictions_matrix = svd_predict(filled_ratings_matrix, k)
        text_result.insert(tk.END, "预测完成。\n")
    except (ValueError, RuntimeError) as e:
        messagebox.showerror("错误", f"训练预测出错: {e}")


def recommend_movies():
    global predictions_matrix
    user_id = int(entry_user_id.get())  # 获取用户输入的要推荐电影的用户ID
    top_n = int(entry_top_n.get())  # 获取用户输入的推荐电影数量
    try:
        recommended_movies = recommend_movies_for_user(user_id, predictions_matrix, ratings_df, top_n)
        text_result.insert(tk.END, f"为用户ID {user_id} 推荐的电影：\n")
        for movie_title, pred in recommended_movies:
            text_result.insert(tk.END, f"电影标题: {movie_title}, 预测评分: {pred}\n")
    except Exception as e:
        messagebox.showerror("错误", f"推荐电影出错: {e}")


root = tk.Tk()
root.title("电影推荐系统")

label_file_path = tk.Label(root, text="请输入数据文件路径：")
label_file_path.pack()
entry_file_path = tk.Entry(root)
entry_file_path.pack()
button_load = tk.Button(root, text="加载数据", command=load_data)
button_load.pack()

label_target_count = tk.Label(root, text="请输入筛选共同评分最多的数量：")
label_target_count.pack()
entry_target_count = tk.Entry(root)
entry_target_count.pack()
button_process = tk.Button(root, text="处理数据", command=process_data)
button_process.pack()

label_k = tk.Label(root, text="请输入SVD分解的k值：")
label_k.pack()
entry_k = tk.Entry(root)
entry_k.pack()
button_train_predict = tk.Button(root, text="训练并预测", command=train_and_predict)
button_train_predict.pack()

label_user_id = tk.Label(root, text="请输入用户ID：")
label_user_id.pack()
entry_user_id = tk.Entry(root)
entry_user_id.pack()
label_top_n = tk.Label(root, text="请输入推荐电影数量：")
label_top_n.pack()
entry_top_n = tk.Entry(root)
entry_top_n.pack()
button_recommend = tk.Button(root, text="推荐电影", command=recommend_movies)
button_recommend.pack()

text_result = tk.Text(root)
text_result.pack()

root.mainloop()