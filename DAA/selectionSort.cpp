#include <bits/stdc++.h>
using namespace std;
void display(vector<int> arr){
    for (int i : arr){
        cout << i << " ";
    } cout << endl;
}
void ssort(vector<int> arr){
    for (int i = 0; i < arr.size(); i++){
        int minimum = i;
        for (int j = i + 1; j < arr.size(); j++){
            if (arr[j] < arr[minimum]){
                minimum = j;
            }
        }
        swap(arr[minimum], arr[i]);
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
    ssort(nums);
}
