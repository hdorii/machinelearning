#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#pragma warning(disable:4996)
#define INPUT_NODES 784
#define HIDDEN_LAYER1_NODES 128
#define HIDDEN_LAYER2_NODES 64
#define OUTPUT_NODES 10
#define INITIAL_LEARNING_RATE 0.001
#define EPOCHS 50
#define LEARNING_RATE_DECAY 0.9
#define MOMENTUM 0.9
#define FIXED_NUM_SAMPLES 60000
#define NUM_CLASSES 10

double relu(double x) {
    return (x > 0) ? x : 0;
}

double relu_derivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}



void softmax(double* output, int length) {
    double max = output[0];
    for (int i = 1; i < length; i++) {
        if (output[i] > max) max = output[i];
    }
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(output[i] - max);  // Max subtraction for numerical stability
        sum += output[i];
    }
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

void log_loss(double loss) {
    static int count = 0;
    printf("Epoch %d, Loss: %.4f\n", count++, loss);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

typedef struct {
    double input_to_hidden1[INPUT_NODES][HIDDEN_LAYER1_NODES];
    double hidden1_to_hidden2[HIDDEN_LAYER1_NODES][HIDDEN_LAYER2_NODES];
    double hidden2_to_output[HIDDEN_LAYER2_NODES][OUTPUT_NODES];
    double* input;

    // Momentum terms
    double input_to_hidden1_momentum[INPUT_NODES][HIDDEN_LAYER1_NODES];
    double hidden1_to_hidden2_momentum[HIDDEN_LAYER1_NODES][HIDDEN_LAYER2_NODES];
    double hidden2_to_output_momentum[HIDDEN_LAYER2_NODES][OUTPUT_NODES];
} NeuralNetwork;

void initialize_weights(NeuralNetwork* nn) {
    double stddev;
    // Initialize weights for the first hidden layer with Xavier initialization
    int nin = INPUT_NODES;
    int nout = HIDDEN_LAYER1_NODES;
    stddev = sqrt(2.0 / (nin + nout));
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_LAYER1_NODES; j++) {
            nn->input_to_hidden1[i][j] = stddev * ((double)rand() / RAND_MAX * 2 - 1);
        }
    }

    // Initialize weights for the second hidden layer with Xavier initialization
    nin = HIDDEN_LAYER1_NODES;
    nout = HIDDEN_LAYER2_NODES;
    stddev = sqrt(2.0 / (nin + nout));
    for (int i = 0; i < HIDDEN_LAYER1_NODES; i++) {
        for (int j = 0; j < HIDDEN_LAYER2_NODES; j++) {
            nn->hidden1_to_hidden2[i][j] = stddev * ((double)rand() / RAND_MAX * 2 - 1);
        }
    }

    // Initialize weights for the output layer with Xavier initialization
    nin = HIDDEN_LAYER2_NODES;
    nout = OUTPUT_NODES;
    stddev = sqrt(2.0 / (nin + nout));
    for (int i = 0; i < HIDDEN_LAYER2_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            nn->hidden2_to_output[i][j] = stddev * ((double)rand() / RAND_MAX * 2 - 1);
        }
    }

    // Initialize momentum terms to zero
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_LAYER1_NODES; j++) {
            nn->input_to_hidden1_momentum[i][j] = 0.0;
        }
    }
    for (int i = 0; i < HIDDEN_LAYER1_NODES; i++) {
        for (int j = 0; j < HIDDEN_LAYER2_NODES; j++) {
            nn->hidden1_to_hidden2_momentum[i][j] = 0.0;
        }
    }
    for (int i = 0; i < HIDDEN_LAYER2_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            nn->hidden2_to_output_momentum[i][j] = 0.0;
        }
    }
}




int initialize_and_read_data(NeuralNetwork* nn, int file_number, int i) {
    char filename[8000];

    sprintf(filename, "../../../mnitst_raw/training/%d/%d-%d.raw", file_number, file_number, i);
    FILE* fp = fopen(filename, "rb"); // fopen으로 변경
    if (fp == NULL) {
        printf("Failed to open file %s\n", filename);
        return 1;
    }

    unsigned char raw_data[784];
    fread(raw_data, sizeof(unsigned char), 784, fp);


    nn->input = (double*)malloc(INPUT_NODES * sizeof(double));
    if (nn->input == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for nn.input\n");
        return 1;
    }
    for (int j = 0; j < INPUT_NODES; j++) {
        nn->input[j] = raw_data[j] / 255.0; // 픽셀 값 정규화
    }

    fclose(fp); // 파일 읽은 후 닫기

    return 0;
}

int correct_predictions[NUM_CLASSES] = { 0 };
int total_predictions[NUM_CLASSES] = { 0 };

void update_accuracy(int predicted, int actual) {
    total_predictions[actual]++;
    if (predicted == actual) {
        correct_predictions[actual]++;
    }
}

void log_accuracies() {
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (total_predictions[i] > 0) {
            printf("Accuracy for %d: %.2f%%\n", i, 100.0 * correct_predictions[i] / total_predictions[i]);
        }
        else {
            printf("No data for digit %d\n", i);
        }
    }
}




int read_test_data(NeuralNetwork* nn, int file_number, int i) {
    char filename[8000];
    sprintf(filename, "../../../mnitst_raw/testing/%d/%d-%d.raw", file_number, file_number, i);
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Failed to open file %s\n", filename);
        return 1;
    }

    unsigned char raw_data[784];
    size_t read_count = fread(raw_data, sizeof(unsigned char), 784, fp);
    if (read_count != 784) { // 데이터 읽기 실패 확인
        printf("Failed to read full data for %s, read %zu bytes.\n", filename, read_count);
        fclose(fp);
        return 1;
    }

    nn->input = (double*)malloc(INPUT_NODES * sizeof(double));
    if (nn->input == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for nn.input\n");
        fclose(fp);
        return 1;
    }
    for (int j = 0; j < INPUT_NODES; j++) {
        nn->input[j] = raw_data[j] / 255.0; // 픽셀 값 정규화
    }

    fclose(fp); // 파일 읽은 후 닫기

    // 성공적으로 테스트 데이터를 읽었다는 로그 메시지 추가
    printf("Successfully read test data from %s\n", filename);

    return 0;
}





void save_metadata(NeuralNetwork* nn, const char* filename) {
    FILE* fp = fopen(filename, "w");  // 메타데이터를 저장할 파일 열기
    if (fp == NULL) {
        fprintf(stderr, "Error opening file for metadata\n");
        return;
    }

    // 은닉층의 개수와 각 은닉층의 노드 수 저장
    fprintf(fp, "Hidden_Layers = %d\n", 2);  // 은닉층은 2개라고 가정
    fprintf(fp, "Hidden_Layer1_Nodes = %d\n", HIDDEN_LAYER1_NODES);
    fprintf(fp, "Hidden_Layer2_Nodes = %d\n", HIDDEN_LAYER2_NODES);

    // 가중치 저장
    // 첫 번째 은닉층 가중치
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_LAYER1_NODES; j++) {
            fprintf(fp, "Weight_Input_Hidden1[%d][%d] = %f\n", i, j, nn->input_to_hidden1[i][j]);
        }
    }

    // 두 번째 은닉층 가중치
    for (int i = 0; i < HIDDEN_LAYER1_NODES; i++) {
        for (int j = 0; j < HIDDEN_LAYER2_NODES; j++) {
            fprintf(fp, "Weight_Hidden1_Hidden2[%d][%d] = %f\n", i, j, nn->hidden1_to_hidden2[i][j]);
        }
    }

    // 출력층 가중치
    for (int i = 0; i < HIDDEN_LAYER2_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            fprintf(fp, "Weight_Hidden2_Output[%d][%d] = %f\n", i, j, nn->hidden2_to_output[i][j]);
        }
    }

    fclose(fp);  // 파일 닫기
}






void forward_pass(NeuralNetwork* nn, double* hidden1, double* hidden2, double* output) {
    // Calculate activations for the first hidden layer using ReLU
    for (int i = 0; i < HIDDEN_LAYER1_NODES; i++) {
        double activation = 0.0;
        for (int j = 0; j < INPUT_NODES; j++) {
            activation += nn->input[j] * nn->input_to_hidden1[j][i];
        }
        hidden1[i] = relu(activation);  // ReLU 함수 적용
    }

    // Calculate activations for the second hidden layer using ReLU
    for (int i = 0; i < HIDDEN_LAYER2_NODES; i++) {
        double activation = 0.0;
        for (int j = 0; j < HIDDEN_LAYER1_NODES; j++) {
            activation += hidden1[j] * nn->hidden1_to_hidden2[j][i];
        }
        hidden2[i] = relu(activation);  // ReLU 함수 적용
    }

    // Calculate activations for the output layer, no ReLU here
    for (int i = 0; i < OUTPUT_NODES; i++) {
        double activation = 0.0;
        for (int j = 0; j < HIDDEN_LAYER2_NODES; j++) {
            activation += hidden2[j] * nn->hidden2_to_output[j][i];
        }
        output[i] = activation;  // No activation function here, will use softmax next
    }

    // Apply softmax to output layer to obtain probabilities
    softmax(output, OUTPUT_NODES);
}


void backpropagation(NeuralNetwork* nn, double* hidden1, double* hidden2, double* output, int target, double* loss) {
    double output_error[OUTPUT_NODES] = { 0 };
    double hidden2_error[HIDDEN_LAYER2_NODES] = { 0 };
    double hidden1_error[HIDDEN_LAYER1_NODES] = { 0 };
    *loss = 0.0;

    // 1. 출력층 오차 계산 (소프트맥스와 크로스 엔트로피 조합)
    for (int i = 0; i < OUTPUT_NODES; i++) {
        double target_value = (i == target) ? 1.0 : 0.0;
        output_error[i] = output[i] - target_value;
        *loss += -target_value * log(output[i] + 1e-8);
    }

    // 2. 두 번째 은닉층 오차 계산 (ReLU 도함수 적용)
    for (int i = 0; i < HIDDEN_LAYER2_NODES; i++) {
        hidden2_error[i] = 0;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            hidden2_error[i] += output_error[j] * nn->hidden2_to_output[i][j];
        }
        hidden2_error[i] *= relu_derivative(hidden2[i]);
    }

    // 3. 첫 번째 은닉층 오차 계산 (ReLU 도함수 적용)
    for (int i = 0; i < HIDDEN_LAYER1_NODES; i++) {
        hidden1_error[i] = 0;
        for (int j = 0; j < HIDDEN_LAYER2_NODES; j++) {
            hidden1_error[i] += hidden2_error[j] * nn->hidden1_to_hidden2[i][j];
        }
        hidden1_error[i] *= relu_derivative(hidden1[i]);
    }

    // 4. 가중치 업데이트
    // 출력층 가중치 업데이트
    for (int i = 0; i < HIDDEN_LAYER2_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            nn->hidden2_to_output[i][j] -= LEARNING_RATE * output_error[j] * hidden2[i];
        }
    }

    // 두 번째 은닉층 가중치 업데이트
    for (int i = 0; i < HIDDEN_LAYER1_NODES; i++) {
        for (int j = 0; j < HIDDEN_LAYER2_NODES; j++) {
            nn->hidden1_to_hidden2[i][j] -= LEARNING_RATE * hidden2_error[j] * hidden1[i];
        }
    }

    // 첫 번째 은닉층 가중치 업데이트
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_LAYER1_NODES; j++) {
            nn->input_to_hidden1[i][j] -= LEARNING_RATE * hidden1_error[j] * nn->input[i];
        }
    }
}



// Evaluate the network's prediction
void evaluate_prediction(double* output, int actual_label) {
    int predicted_label = 0;
    double max_prob = output[0];
    for (int i = 1; i < OUTPUT_NODES; i++) {
        if (output[i] > max_prob + 0.5) {
            max_prob = output[i];
            predicted_label = i;
        }
    }

    printf("Predicted label: %d, Actual label: %d, Confidence: %.2f%%\n",
        predicted_label, actual_label, max_prob * 100);

    update_accuracy(predicted_label, actual_label);
}

int main() {
    NeuralNetwork nn;
    double hidden1[HIDDEN_LAYER1_NODES], hidden2[HIDDEN_LAYER2_NODES], output[OUTPUT_NODES];
    int correct_predictions_per_number[NUM_CLASSES] = { 0 };
    int total_tests_per_number[NUM_CLASSES] = { 0 };

    // 가중치 초기화
    initialize_weights(&nn);
    double current_learning_rate = INITIAL_LEARNING_RATE;
    // 훈련 과정
    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        current_learning_rate *= LEARNING_RATE_DECAY;
        for (int file_number = 0; file_number <= 9; file_number++) {
            for (int i = 0; i < 5000; i++) {
                nn.input = (double*)malloc(INPUT_NODES * sizeof(double));
                if (nn.input == NULL) {
                    fprintf(stderr, "Error: Failed to allocate memory for nn.input\n");
                    return 1;
                }
                if (initialize_and_read_data(&nn, file_number, i) != 0) {
                    free(nn.input);  // 실패 시 메모리 해제
                    return 1;
                }
                double loss = 0.0;
                forward_pass(&nn, hidden1, hidden2, output);
                backpropagation(&nn, hidden1, hidden2, output, file_number, &loss);
                log_loss(loss);
                free(nn.input);  // 메모리 해제
            }
        }
    }

    // 모델 메타데이터 저장
    save_metadata(&nn, "neural_network_final_metadata.txt");

    // 테스트 데이터로 정확도 측정
    for (int file_number = 0; file_number < NUM_CLASSES; file_number++) {
        for (int i = 0; i < 800; i++) {
            nn.input = (double*)malloc(INPUT_NODES * sizeof(double));
            if (nn.input == NULL) {
                fprintf(stderr, "Error: Failed to allocate memory for nn.input\n");
                free(nn.input);
                return 1;
            }
            if (read_test_data(&nn, file_number, i) != 0) {
                free(nn.input);
                return 1;
            }
            forward_pass(&nn, hidden1, hidden2, output);
            evaluate_prediction(output, file_number);  // 결과 평가
            free(nn.input);
        }
    }

    // 전체 정확도 출력
    log_accuracies();

    return 0;
}