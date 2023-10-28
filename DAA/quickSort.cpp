#include <bits/stdc++.h>
using namespace std;
void display(vector<int> arr){
    for (int i : arr){
        cout << i << " ";
    } cout << endl;
}
int partition(vector<int> &arr, int l, int r){
    int pivot = arr[l];
    int count = 0;
    for (int i = l + 1; i <= r; i++){
        if (arr[i] <= pivot){
            count++;
        }
    }
    int p = l + count;
    swap(arr[p], arr[l]);
    
    int i = l, j = r;
    while (i < p && j > p){
        while (arr[i] <= pivot) i++;
        while (arr[j] > pivot) j--;
        if (i < p && j > p){
            swap(arr[i], arr[j]);
            i++;
            j--;
        }
    }
    return p;
}
void quickSort(vector<int> &arr, int l, int r){
    if (l >= r){
        return;
    }
    int p = partition(arr, l, r);
    
    quickSort(arr, l, p - 1);
    quickSort(arr, p + 1, r);
}
int main(){
    cout << "Shivesh Ranjan 035" << endl;
    vector<int> arr = {3, 4, 2, 7, 9, 11, 1, 5};
    cout << "Array to be sorted:" << endl;
    display(arr);
    cout << endl << "Sorted Array:" << endl;
    quickSort(arr, 0, arr.size() - 1);
    display(arr);
}
