#include <bits/stdc++.h>
using namespace std;
void display(vector<int> arr){
    for (int i : arr){
        cout << i << " ";
    } cout << endl;
}
void merge(vector<int> &arr, int l, int r, int mid){
    int n1 = l;
    int n2 = mid + 1;
    vector<int> temp;
    while (n1 <= mid && n2 <= r){
        if (arr[n1] < arr[n2]){
            temp.push_back(arr[n1]);
            n1++;
        }
        else{
            temp.push_back(arr[n2]);
            n2++;
        }
    }
    while (n1 <= mid){
        temp.push_back(arr[n1]);
        n1++;
    }
    while (n2 <= r){
        temp.push_back(arr[n2]);
        n2++;
    }
    for (int i = l; i <= r; i++){
        arr[i] = temp[i - l];
    }
    display(arr);
}
void msort(vector<int>& arr, int l, int r){
    if (l >= r){
        return;
    }
    int mid = (l + r) / 2;
    msort(arr, l, mid);
    msort(arr, mid + 1, r);
    merge(arr, l, r, mid);
}
int main(){
    cout << "Shivesh Ranjan 035" << endl;
    vector <int> nums = {5, 4, 2, 3, 7, 8, 1, 2};
    cout << "Array to be sorted:" << endl;
    display(nums);
    cout << endl;
    msort(nums, 0, nums.size() - 1);
    cout << endl << "Sorted Array:" << endl;
    display(nums);
}
