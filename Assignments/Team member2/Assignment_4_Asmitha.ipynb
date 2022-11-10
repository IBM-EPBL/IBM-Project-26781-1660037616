{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "# Import the necessary libraries\n"
      ],
      "metadata": {
        "id": "l__fept8YAa1"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "KJDUmdasTiw2"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from keras import utils\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from keras.models import Model\n",
        "from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding\n",
        "from keras.optimizers import RMSprop\n",
        "from keras.preprocessing.text import Tokenizer\n",
        "from keras.preprocessing import sequence\n",
        "from keras.utils import to_categorical\n",
        "%matplotlib inline\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Preprocessing"
      ],
      "metadata": {
        "id": "pP6Rq_WwYFSR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('/content/spam.csv',delimiter=',',encoding='latin-1')\n",
        "df"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "sZ9ZZoucTsMf",
        "outputId": "0748bc38-693b-4deb-8c96-a541268dc31e"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        v1                                                 v2 Unnamed: 2  \\\n",
              "0      ham  Go until jurong point, crazy.. Available only ...        NaN   \n",
              "1      ham                      Ok lar... Joking wif u oni...        NaN   \n",
              "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN   \n",
              "3      ham  U dun say so early hor... U c already then say...        NaN   \n",
              "4      ham  Nah I don't think he goes to usf, he lives aro...        NaN   \n",
              "...    ...                                                ...        ...   \n",
              "5567  spam  This is the 2nd time we have tried 2 contact u...        NaN   \n",
              "5568   ham              Will Ì_ b going to esplanade fr home?        NaN   \n",
              "5569   ham  Pity, * was in mood for that. So...any other s...        NaN   \n",
              "5570   ham  The guy did some bitching but I acted like i'd...        NaN   \n",
              "5571   ham                         Rofl. Its true to its name        NaN   \n",
              "\n",
              "     Unnamed: 3 Unnamed: 4  \n",
              "0           NaN        NaN  \n",
              "1           NaN        NaN  \n",
              "2           NaN        NaN  \n",
              "3           NaN        NaN  \n",
              "4           NaN        NaN  \n",
              "...         ...        ...  \n",
              "5567        NaN        NaN  \n",
              "5568        NaN        NaN  \n",
              "5569        NaN        NaN  \n",
              "5570        NaN        NaN  \n",
              "5571        NaN        NaN  \n",
              "\n",
              "[5572 rows x 5 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-9b53ce65-70af-4e20-87e4-b75310e1cb43\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>v1</th>\n",
              "      <th>v2</th>\n",
              "      <th>Unnamed: 2</th>\n",
              "      <th>Unnamed: 3</th>\n",
              "      <th>Unnamed: 4</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>ham</td>\n",
              "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>ham</td>\n",
              "      <td>Ok lar... Joking wif u oni...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>spam</td>\n",
              "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>ham</td>\n",
              "      <td>U dun say so early hor... U c already then say...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>ham</td>\n",
              "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5567</th>\n",
              "      <td>spam</td>\n",
              "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5568</th>\n",
              "      <td>ham</td>\n",
              "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5569</th>\n",
              "      <td>ham</td>\n",
              "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5570</th>\n",
              "      <td>ham</td>\n",
              "      <td>The guy did some bitching but I acted like i'd...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5571</th>\n",
              "      <td>ham</td>\n",
              "      <td>Rofl. Its true to its name</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5572 rows × 5 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-9b53ce65-70af-4e20-87e4-b75310e1cb43')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-9b53ce65-70af-4e20-87e4-b75310e1cb43 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-9b53ce65-70af-4e20-87e4-b75310e1cb43');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)\n",
        "df"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 423
        },
        "id": "3moGtNUeTuwO",
        "outputId": "b578248e-91bd-45a4-bf7d-acd8744b4e22"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        v1                                                 v2\n",
              "0      ham  Go until jurong point, crazy.. Available only ...\n",
              "1      ham                      Ok lar... Joking wif u oni...\n",
              "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
              "3      ham  U dun say so early hor... U c already then say...\n",
              "4      ham  Nah I don't think he goes to usf, he lives aro...\n",
              "...    ...                                                ...\n",
              "5567  spam  This is the 2nd time we have tried 2 contact u...\n",
              "5568   ham              Will Ì_ b going to esplanade fr home?\n",
              "5569   ham  Pity, * was in mood for that. So...any other s...\n",
              "5570   ham  The guy did some bitching but I acted like i'd...\n",
              "5571   ham                         Rofl. Its true to its name\n",
              "\n",
              "[5572 rows x 2 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-f2d8b8ef-e002-4490-878f-c6783b387642\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>v1</th>\n",
              "      <th>v2</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>ham</td>\n",
              "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>ham</td>\n",
              "      <td>Ok lar... Joking wif u oni...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>spam</td>\n",
              "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>ham</td>\n",
              "      <td>U dun say so early hor... U c already then say...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>ham</td>\n",
              "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5567</th>\n",
              "      <td>spam</td>\n",
              "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5568</th>\n",
              "      <td>ham</td>\n",
              "      <td>Will Ì_ b going to esplanade fr home?</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5569</th>\n",
              "      <td>ham</td>\n",
              "      <td>Pity, * was in mood for that. So...any other s...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5570</th>\n",
              "      <td>ham</td>\n",
              "      <td>The guy did some bitching but I acted like i'd...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5571</th>\n",
              "      <td>ham</td>\n",
              "      <td>Rofl. Its true to its name</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5572 rows × 2 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f2d8b8ef-e002-4490-878f-c6783b387642')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f2d8b8ef-e002-4490-878f-c6783b387642 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f2d8b8ef-e002-4490-878f-c6783b387642');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.countplot(df.v1,palette='Set3')\n",
        "plt.xlabel('Label')\n",
        "plt.title('Number of ham and spam messages')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 368
        },
        "id": "zEbpqGViVwef",
        "outputId": "6c6bfd2b-da79-431e-e7bd-8306cfd8df80"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  FutureWarning\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'Number of ham and spam messages')"
            ]
          },
          "metadata": {},
          "execution_count": 5
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "X = df.v2\n",
        "Y = df.v1\n",
        "le = LabelEncoder()\n",
        "Y = le.fit_transform(Y)\n",
        "Y = Y.reshape(-1,1)"
      ],
      "metadata": {
        "id": "_tBg2zByVyir"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Split into training and test data."
      ],
      "metadata": {
        "id": "3VAQxQYBYKVz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)"
      ],
      "metadata": {
        "id": "sWDUQRpDV4uu"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "max_words = 1000\n",
        "max_len = 150\n",
        "tok = Tokenizer(num_words=max_words)\n",
        "tok.fit_on_texts(X_train)\n",
        "sequences = tok.texts_to_sequences(X_train)\n",
        "sequences_matrix = utils.pad_sequences(sequences,maxlen=max_len)"
      ],
      "metadata": {
        "id": "01SI0QpCV6hx"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sequences_matrix.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aP6ZEbgeV87b",
        "outputId": "a645f8d9-b336-4da0-c3ec-c5ccd1919079"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(4736, 150)"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sequences_matrix.ndim"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6LbhZkkmV_G7",
        "outputId": "bf779cd4-6336-4617-a033-ca845f116a03"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sequences_matrix = np.reshape(sequences_matrix,(4736,150,1))"
      ],
      "metadata": {
        "id": "Gjs8zJ0UWA3b"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sequences_matrix.ndim"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8fXSlfxzWCPv",
        "outputId": "f00ad16a-c971-47cd-9fdd-83f950f754ed"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "3"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Create model for RNN"
      ],
      "metadata": {
        "id": "XYzPDVmeYPrO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import Dense\n",
        "from keras.layers import LSTM\n",
        "from keras.layers import Embedding"
      ],
      "metadata": {
        "id": "vK-8XSL0WEDb"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = Sequential()"
      ],
      "metadata": {
        "id": "e7ctzwEeWlqs"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Add Layers"
      ],
      "metadata": {
        "id": "9pHgiRD9YTNV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.add(Embedding(max_words,50,input_length=max_len))\n",
        "model.add(LSTM(units=64,input_shape = (sequences_matrix.shape[1],1),return_sequences=True))\n",
        "model.add(LSTM(units=64,return_sequences=True))\n",
        "model.add(LSTM(units=64,return_sequences=True))\n",
        "model.add(LSTM(units=64))\n",
        "model.add(Dense(units = 256,activation = 'relu'))\n",
        "model.add(Dense(units = 1,activation = 'sigmoid'))"
      ],
      "metadata": {
        "id": "z-NmquktWJ-P"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Compile the model"
      ],
      "metadata": {
        "id": "Fd1KkcZ4YXZd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.summary()\n",
        "model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6lt8VSdZWiED",
        "outputId": "0ca11081-75ca-4dee-9507-5a697f207f3f"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " embedding (Embedding)       (None, 150, 50)           50000     \n",
            "                                                                 \n",
            " lstm (LSTM)                 (None, 150, 64)           29440     \n",
            "                                                                 \n",
            " lstm_1 (LSTM)               (None, 150, 64)           33024     \n",
            "                                                                 \n",
            " lstm_2 (LSTM)               (None, 150, 64)           33024     \n",
            "                                                                 \n",
            " lstm_3 (LSTM)               (None, 64)                33024     \n",
            "                                                                 \n",
            " dense (Dense)               (None, 256)               16640     \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 1)                 257       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 195,409\n",
            "Trainable params: 195,409\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Fit the model on the training data."
      ],
      "metadata": {
        "id": "GR1va4sTYc68"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "M = model.fit(sequences_matrix,Y_train,batch_size=128,epochs=5,validation_split=0.2)\n",
        "M"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xJVlS7ElWrEz",
        "outputId": "b655d55f-3145-4d0e-e996-1656ef3200f0"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "30/30 [==============================] - 26s 678ms/step - loss: 0.4399 - accuracy: 0.8730 - val_loss: 0.4236 - val_accuracy: 0.8513\n",
            "Epoch 2/5\n",
            "30/30 [==============================] - 19s 645ms/step - loss: 0.3632 - accuracy: 0.8728 - val_loss: 0.3418 - val_accuracy: 0.8513\n",
            "Epoch 3/5\n",
            "30/30 [==============================] - 19s 628ms/step - loss: 0.1497 - accuracy: 0.9475 - val_loss: 0.0908 - val_accuracy: 0.9747\n",
            "Epoch 4/5\n",
            "30/30 [==============================] - 19s 649ms/step - loss: 0.0653 - accuracy: 0.9826 - val_loss: 0.0735 - val_accuracy: 0.9821\n",
            "Epoch 5/5\n",
            "30/30 [==============================] - 19s 626ms/step - loss: 0.0385 - accuracy: 0.9905 - val_loss: 0.0704 - val_accuracy: 0.9821\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7f95f6cf2b10>"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Save the model"
      ],
      "metadata": {
        "id": "Ae3r-Ii1YqHS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.save"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lpD_OzsJWuNe",
        "outputId": "1d5b5ce0-50d4-4974-b781-ad4466e74ce1"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<bound method Model.save of <keras.engine.sequential.Sequential object at 0x7f95fb1f4f90>>"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Evaluate the model on test set data."
      ],
      "metadata": {
        "id": "dQpS-lu-YtWJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "test_sequences = tok.texts_to_sequences(X_test)\n",
        "test_sequences_matrix = utils.pad_sequences(test_sequences,maxlen=max_len)"
      ],
      "metadata": {
        "id": "IFmPKxgDXsKb"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "accr = model.evaluate(test_sequences_matrix,Y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TIlwu-l9XvhT",
        "outputId": "fa709ea9-47bf-487a-9d3f-ce9217d4848e"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "27/27 [==============================] - 3s 53ms/step - loss: 0.1055 - accuracy: 0.9701\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "l = accr[0]\n",
        "a =accr[1]\n",
        "print('Test set\\n  Loss: {:0.3f}\\n  Accuracy: {:0.3f}'.format(l,a))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LFFQpdUiXw33",
        "outputId": "7953da36-6e44-43af-c9c9-a9d76eee9eff"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Test set\n",
            "  Loss: 0.105\n",
            "  Accuracy: 0.970\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Accuracy and Loss Graph"
      ],
      "metadata": {
        "id": "j3rRqpBIYwdb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "results = pd.DataFrame({\"Train Loss\": M.history['loss'], \"Validation Loss\": M.history['val_loss'],\n",
        "              \"Train Accuracy\": M.history['accuracy'], \"Validation Accuracy\": M.history['val_accuracy']\n",
        "             })\n",
        "fig, ax = plt.subplots(nrows=2, figsize=(16, 9))\n",
        "results[[\"Train Loss\", \"Validation Loss\"]].plot(ax=ax[0])\n",
        "results[[\"Train Accuracy\", \"Validation Accuracy\"]].plot(ax=ax[1])\n",
        "ax[0].set_xlabel(\"Epoch\")\n",
        "ax[1].set_xlabel(\"Epoch\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 551
        },
        "id": "pJ3RiE8XXz-a",
        "outputId": "b5bb7fda-d01c-44e9-c501-de0a850fe839"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1152x648 with 2 Axes>"
            ],
            "image/png": "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ehERkSDTNagiIiLtzH8Nara19kCwaxEREQll6uIrIiIiIiIiIUEtqCIiIiIiIhIS1IIqIiIiIiIiIUEBVUREREREREKCAqqIiIiIiIiEBAVUERERERERCQkKqCIiIiIiIhISFFBFREREREQkJCigioiIiIiISEhQQBUREREREZGQoIAqIiIiIiIiIUEBVUREREREREKCAqqIiIiIiIiEhPBgF9BYz549bUZGRrDLEBERERERkXawfv36A9ba1KbWhVxAzcjIIDc3N9hliIiIiIiISDswxnzd3Dp18RUREREREZGQoIAqIiIiIiIiIUEBVUREREREREJCyF2D2hSXy0VBQQFVVVXBLkVCiNPpJC0tjYiIiGCXIiIiIiIibaBTBNSCggLi4+PJyMjAGBPsciQEWGspLCykoKCAzMzMYJcjIiIiIiJtoFN08a2qqiIlJUXhVOoYY0hJSVGruoiIiIhIF9IpAiqgcCpH0L8JEREREZGupVN08RUREREREZEjWWs5VOOhvMpNWZWLsmo3aUnR9EpwBru0Y6KAGoDCwkIuuOACAPbs2YPD4SA1NRWAtWvXEhkZ2exzc3NzmTdvHk8++WSrjrlx40ZGjBjBsmXLuPjii4+9eBERERERCUnVbg9lVW5/uHRTVu2q99hFeXXtcrd/uW99WZWb8mo3pVUuDlW78dqG+31gwincODYjKK/peCmgBiAlJYWNGzcCcP/99xMXF8ddd91Vt97tdhMe3vSvMjs7m+zs7FYfc8GCBZx99tksWLCgXQOqx+PB4XC02/5FRERERLoaj9f6QmRtoKz2t17WC49lVa56wbNh4KxdXuPxtnisyPAwEpzhxEWFE++MIC4qnAEpMcQ5w0nwP453hhPn9K2PjwpnaO/4DvgttA8F1GM0ffp0nE4nH3/8MWeddRZTp07lzjvvpKqqiujoaJ577jmGDh3KqlWrmD17Nv/85z+5//772bVrF3l5eezatYtZs2Zxxx13HLFvay2LFi1ixYoVnHPOOVRVVeF0+proH330UV566SXCwsK45JJLeOSRR9i+fTu33nor+/fvx+FwsGjRIvLz8+uOC3D77beTnZ3N9OnTycjIYMqUKaxYsYJf/OIXlJWVMXfuXGpqahg0aBAvvvgiMTEx7N27l1tvvZW8vDwA5syZw/Lly0lOTmbWrFkA/OpXv6JXr17c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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "HfktCcpdX3AP"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}