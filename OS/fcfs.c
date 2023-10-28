#include <stdio.h>
void findWaitingTime(int processes[], int n, int bt[], int wt[]) {
  // waiting time for first process is 0
  wt[0] = 0;
  // calculating waiting time
  for (int i = 1; i < n; i++)
    wt[i] = bt[i - 1] + wt[i - 1];
}
// Function to calculate turn around time
void findTurnAroundTime(int processes[], int n, int bt[], int wt[], int tat[]) {
  // calculating turnaround time by adding
  // bt[i] + wt[i]
  for (int i = 0; i < n; i++)
    tat[i] = bt[i] + wt[i];
}
// Function to calculate average time
void findavgTime(int processes[], int n, int bt[]) {
  int wt[n], tat[n], total_wt = 0, total_tat = 0;
  // Function to find waiting time of all processes
  findWaitingTime(processes, n, bt, wt);
  // Function to find turn around time for all processes
  findTurnAroundTime(processes, n, bt, wt, tat);
  // Display processes along with all details
  printf("Process |  Burst time\t|  Waiting time\t|  Turn around time\n");
  for (int i = 0; i < 60; i++)
    printf("-");
  printf("\n");
  // Calculate total waiting time and total turn
  for (int i = 0; i < n; i++) {
    total_wt = total_wt + wt[i];
    total_tat = total_tat + tat[i];
    printf(" %d ", (i + 1));
    printf("\t|\t%d ", bt[i]);
    printf("\t|\t%d", wt[i]);
    printf("\t|\t%d\n", tat[i]);
  }
  float s = (float)total_wt / (float)n;
  float t = (float)total_tat / (float)n;
  printf("Average waiting time = %f", s);
  printf("\n");
  printf("Average turn around time = %f \n", t);
}
int main() {
  printf("Shivesh 035\n");
  int processes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int n = sizeof processes / sizeof processes[0];

  int burstTime[] = {10, 5, 8, 4, 3, 6, 4, 6, 3, 10};
  findavgTime(processes, n, burstTime);
  return 0;
}
