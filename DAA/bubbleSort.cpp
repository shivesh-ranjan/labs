#include <bits/stdc++.h>
using namespace std;
void display(vector<int> arr){
    for (int i : arr){
        cout << i << " ";
    } cout << endl;
}
void bsort(vector<int> arr){
    int n = arr.size();
    int i = 0;
    int j = n - 1;
    for (int i = 0; i < n; i++){
        bool swapped = false;
        for (int j = 1; j < n - i; j++){
            if (arr[j] < arr[j - 1]){
                swap(arr[j], arr[j - 1]);
                swapped = true;
            }
        }
        if (!swapped) return;
        cout << i << "th iteration: ";
        display(arr);
    }
}
int main(){
    cout<< "Shivesh Ranjan 035" <<endl;
    vector <int> nums = {5, 4, 2, 3, 7, 8, 1, 2};
    cout << "Array to be sorted:" << endl;
    display(nums);
    cout << endl;
    bsort(nums);
}
