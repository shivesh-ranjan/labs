#include <iostream>
using namespace std;

void selectionSort(int *arr, int n) {
  // sorted arr from left side!
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      if (arr[j] < arr[i]) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
  }
}

void bubbleSort(int *arr, int n) {
  // sorted arr from right side! Decreasing at the end
  int counter = 1;
  while (counter < n) {
    for (int i = 0; i < n - counter; i++) {
      if (arr[i] > arr[i + 1]) {
        int temp = arr[i];
        arr[i] = arr[i + 1];
        arr[i + 1] = temp;
      }
    }
    counter++;
  }
}

void insertionSort(int *arr, int n) {
  for (int i = 1; i < n; i++) {
    int current = arr[i];
    int j = i - 1;
    while (arr[j] > current && j >= 0) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = current;
  }
}

void printArray(int *arr, int n) {
  for (int i = 0; i < n; i++) {
    cout << arr[i] << " ";
  }
  cout << endl;
}

int main() {
  int n;
  // cin >> n;
  // int arr[n];
  // cout << "Enter the elements:";
  n = 6;
  int arr[] = {12, 45, 23, 51, 19, 8};
  // for (int i = 0; i < n; i++) {
  //   cin >> arr[i];
  // }
  // selectionSort(&arr[0], n);
  // bubbleSort(&arr[0], n);
  insertionSort(&arr[0], n);
  printArray(&arr[0], n);
}
