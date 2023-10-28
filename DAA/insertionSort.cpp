#include <bits/stdc++.h>
using namespace std;
void display(vector<int> arr){
    for (int i : arr){
        cout << i << " ";
    } cout << endl;
}
void isort(vector<int> arr){
    for (int i = 1; i < arr.size(); i++){
        int pivot = arr[i];
        int j = i - 1;
        while (j >= 0 && pivot < arr[j]){
            arr[j + 1] = arr[j];
            j--;
        }
        arr[++j] = pivot;
        cout << i << "th iteration: ";
        display(arr);
    }
}
int main(){
    cout << "Shivesh Ranjan 035" << endl;
    vector <int> nums = {5, 4, 2, 3, 7, 8, 1, 2};
    cout << "Array to be sorted:" << endl;
    display(nums);
    cout << endl;
    isort(nums);
}
