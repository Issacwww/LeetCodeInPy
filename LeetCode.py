class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
    def __repr__(self):
        return "["+str(self.start)+","+str(self.end)+"]"
class solution:
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        l1, l2 = len(word1), len(word2)
        if l1 == 0 or l2 == 0:
            return max(l1,l2)        
        dp = [[0 for i in range(l1+1)] for j in range(l2+1)]
        for i in range(l2+1):
            dp[i][0] = i
        for j in range(l1+1):
            dp[0][j] = j
        for i in range(1, l2+1):
            for j in range(1, l1+1):
                if(word1[j-1] == word2[i-1]):
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+1)
        return dp[-1][-1]
        # words = ["distance","springbok"]
        # print(solution.minDistance(words[0], words[1]))
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        if not triangle:
            return 0
        size = len(triangle)
        for i in range(1, size):
            for j in range(i+1):
                if j == 0:
                    triangle[i][j] = triangle[i][j] + triangle[i-1][j]
                elif j == i:
                    triangle[i][j] = triangle[i][j] + triangle[i-1][j-1]
                else:
                    triangle[i][j] = triangle[i][j]+min(triangle[i-1][j-1],triangle[i-1][j])
        return triangle
        #triangle = [[2],[3,4],[6,5,7]]
        #print(solution.minimumTotal(triangle))

    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        nums, target = [1, 0, -1, 0, -2, 2], 0
        """
        def findNsum(l, r, target, N, result, results):
            if r-l+1 < N or N < 2 or target < nums[l]*N or target > nums[r]*N:  # early termination
                return
            if N == 2: # two pointers solve sorted 2-sum problem
                while l < r:
                    s = nums[l] + nums[r]
                    if s == target:
                        results.append(result + [nums[l], nums[r]])
                        l += 1
                        while l < r and nums[l] == nums[l-1]:
                            l += 1
                    elif s < target:
                        l += 1
                    else:
                        r -= 1
            else: # recursively reduce N
                for i in range(l, r+1):
                    if i == l or (i > l and nums[i-1] != nums[i]):
                        findNsum(i+1, r, target-nums[i], N-1, result+[nums[i]], results)
        nums.sort()
        results = []
        findNsum(0, len(nums)-1, target, 4, [], results)
        return results


    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        # nums = [2,1,3,4]
        """
        nums.append(0)
        l = len(nums)
        for i in range(l):
            if nums[i] < 0 or nums[i] > l:
                nums[i] = 0
        for i in range(l):
            nums[nums[i] % l ] += l
        for i in range( l):
            if nums[i] // l == 0:
                return i
        return l
    
    def firstMissingPositive2(self, nums):
        if not (nums):
            return 1
        i = 1
        while i in nums:
            i += 1
        return i

    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        [1],1
        """
        def lsize(head):
            size = 1
            while head.next:
                size += 1
                head = head.next
            return size
        print(lsize(head))
        pos = lsize(head) - n + 1
        pre, tmp = None, head
        for i in range(pos):
            pre = tmp
            tmp = tmp.next
        pre.next = tmp.next
        return head

    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        
        not so good
        if not nums:
            return 0
        if len(nums) <= 2:
            return max(nums)
        dp = [0]*len(nums)
        dp[0], dp[1] = nums[0], max(nums[0], nums[1]) 
        for i in range(2, len(nums)):
            dp[i] = max(nums[i] + max(dp[:i-1]),dp[i-1])
        return dp[-1]
        """

        '''
        const extra space
        '''
        rob, nrob = 0,0
        for i in range(len(nums)):
            cur = nrob + nums[i]
            nrob = max(nrob, rob)
            rob = cur
        return max(rob, nrob)

    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        dummy = jump = ListNode(0)
        dummy.next = l = r = head
        
        while 1:
            cnt = 0
            while r and cnt < k:
                cnt += 1
                r = r.next 
            if cnt == k:
                pre, cur = r, l
                for _ in range(k):
                    cur.next, cur, pre = pre, cur.next, cur
                #r = cur
                jump.next, jump, l = pre, l, r 
            else: 
                return dummy.next
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        num = 0
        for d in digits:
            num = num*10+d
        num += 1
        print(num)
        res = []
        while num > 0:
            d = num % 10
            res = [d] + res
            num = num // 10
        return res
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        res = [0]*(rowIndex+1)
        res[0] = 1
        for i in range(1,rowIndex+1):
            for j in range(i,0,-1):
                res[j] += res[j-1]
        return res
    def generate(self, numRows):
        res = [[1]]
        for i in range(1, numRows):
            res += [list(map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1]))]
            #learn how this function work
        return res[:numRows]

    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m-=1
            else:
                nums1[m+n-1] = nums2[n-1]
                n-=1
        if m == 0:
            nums1[:n] = nums2[:n]
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # Solution1
        # nums.sort()
        # if not nums:
        #     return False
        # n = nums[0]
        # for i in range(1,len(nums)):
        #     if nums[i] == n:
        #         return True
        #     n = nums[i]
        # return False
        
        # Solution2
        if not nums:
            return False
        return len(nums) != len(set(nums))
        
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        # Theoretical feasible solution
        
        size = len(nums)
        if k >= size:
            return size != len(set(nums))
        
        hashSet = set(nums[:k])
        if len(hashSet) < k:
            return True
        
        for i in range(k,size):
            hashSet.add(nums[i])
            if len(hashSet)==k:
                return True
            else:
                hashSet.remove(nums[i-k])
        return False

    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return [1]
        res = set(range(1,len(nums)+1))
        for num in set(nums):
            res.remove(num)
        return list(res)

    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        def reverse(alist):
            left = 0
            right = len(alist) - 1
            while left < right:
                temp = alist[left]
                alist[left] = alist[right]
                alist[right] = temp
                
                left += 1
                right -= 1
            return alist
        
        if not nums or len(nums) == 0:
            return
        smallest,smaller = -1,-1
        for i in range(len(nums) - 2,-1,-1):
            if nums[i] < nums[i+1]:
                smallest = i
                break
        if smallest == -1:
            reverse(nums)
            return
        for i in range(len(nums)-1,smallest,-1):
            if nums[i] > nums[smallest]:
                smaller = i
                break
        temp = nums[smallest]
        nums[smallest] = nums[smaller]
        nums[smaller] = temp
        reverse(nums[smallest+1:])

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        size = len(nums)
        if not nums or size == 0:
            return -1
        if size == 1:
            if target == nums[0]:
                return 0
            else:
                return -1
        s = l = r= -1
        for i in range(1,size):
            if nums[i] < nums[i-1]:
                s = i
                break
        if (target < nums[s] or target > nums[s-1]) or (target < nums[0] and target > nums[-1]):
            return -1
        if target >= nums[0]:
            l, r = 0,s-1
        elif target <= nums[-1]:
            l, r = s, size -1
            
        mid = (l + r) // 2
        while l <= r:
            if target == nums[mid]:
                return mid
            elif target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
            mid = (l + r) // 2

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        size = len(nums)
        if not nums or nums == 0:
            return [-1,-1]
        left, right = 0, size - 1
        while nums[left] < nums[right]:
            mid = (left + right) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                left = right = mid
                while left >= 0 and nums[left] == target:
                        left -= 1
                while right <= size-1 and nums[right] == target:
                        right += 1
                return [left+1,right-1]
        return [-1, -1]
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(idx,remain,combo,res):
            if remain == 0:
                res.append(combo)
                return
            if remain < 0:
                return
            for i in range(idx,len(candidates)):
                dfs(i,remain - candidates[i],combo+[candidates[i]],res)
            
        if not candidates or len(candidates) == 0:
            return []
        res =  []
        candidates.sort()
        dfs(0,target,[],res)
        return res
    def spiralOrder(self, matrix):
        return matrix and [*matrix.pop(0)] + self.spiralOrder([*zip(*matrix)][::-1])
    def mergeInter(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        res = []
        for interval in intervals:
            if len(res) == 0:
                res.append(interval)
            else:
                for inter in res:
                    if inter.end >= interval.start:
                        if inter.end < interval.end:
                            inter.end = interval.end
                        break
                    res.append(interval)
                    break
        return res
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        # res = [[0]*n for i in range(n)]
        # num = 2
        # res[0][0] = 1
        # i,j,k = 0,0,0
        # x = [0,1,0,-1]
        # y = [1,0,-1,0]
        # while num <= n**2:
        #     while i+x[k] >= 0 and i+x[k] < n and j+y[k] >=0 and j+y[k] < n and res[i+x[k]][j+y[k]] == 0:
        #         i+=x[k]
        #         j+=y[k]
        #         res[i][j] = num
        #         num+=1  
        #     k = (k+1)%4
        # return res
        if n == 1:
            return [[1]]
        
        matrix = [[0]*n for _ in range(n)]
            
        bcol, brow = 0, 0
        ecol, erow = n-1, n-1
        i = 1
        
        while bcol <= ecol and brow <= erow: 
            
            for j in range(bcol, ecol+1):
                matrix[brow][j] = i 
                i += 1
            brow += 1
            
            for k in range(brow, erow+1):
                matrix[k][ecol] = i 
                i += 1
            ecol -= 1
            
            if brow <= erow: 
                for k in range(ecol, bcol-1, -1):
                    matrix[erow][k] = i 
                    i += 1
                erow -= 1
        
            if bcol <= ecol: 
                for k in range(erow, brow-1, -1):
                    matrix[k][bcol] = i 
                    i += 1
                bcol += 1
    
        return matrix

    def setZeroes(self, matrix: 'List[List[int]]') -> 'None':
        """
        Do not return anything, modify matrix in-place instead.
        """
        m,n = len(matrix),len(matrix[0])
        row = set()
        col = set()
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    row.add(i)
                    col.add(j)
        for i in row:
            matrix[i][:] = [0]*n
        for j in col:
            for i in range(m):
                matrix[i][j] = 0

    