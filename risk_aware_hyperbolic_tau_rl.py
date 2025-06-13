# -*- coding: utf-8 -*-
'''
الأفكار الرئيسة المبتكرة للمطور: باسل يحيى عبدالله
'''
# تم الرفع بتاريخ: 1/5/2025
"""
==============================================================================
 نظام تعلم معزز مبتكر لحل مشكلة GridWorld مع تجنب المخاطر
==============================================================================

الملخص:
هذا الكود يطبق وكيل تعلم معزز (Reinforcement Learning Agent) لحل مشكلة التنقل
في بيئة شبكية (GridWorld) مع عقبات. يتميز الوكيل باستخدام مكونات مبتكرة
تهدف إلى تحسين الأداء، الاستقرار، والأمان مقارنة بالخوارزميات التقليدية.

الأفكار المبتكرة الرئيسية المطبقة:
1.  **طبقة الارتباط الديناميكي (Dynamic Correlation Layer):**
    بدلاً من الاعتماد على تشفير بسيط للحالة، تستخدم هذه الطبقة مصفوفة ارتباط
    (covariance-like matrix) قابلة للتعلم لاكتشاف وتضخيم العلاقات الديناميكية
    وغير الخطية بين ميزات الحالة المختلفة، مما قد يساعد في فهم أعمق لسياق البيئة.

2.  **الانتباه الزائدي (Hyperbolic Attention):**
    تستفيد هذه الآلية من الهندسة الزائدية (Hyperbolic Geometry)، التي يُعتقد أنها
    أكثر ملاءمة لتمثيل البيانات ذات البنية الهرمية (مثل العلاقات المكانية أو
    السببية في مسار الوكيل). يتم حساب أوزان الانتباه بناءً Sعلى المسافات في
    فضاء بوانكاريه الزائدي (Poincaré Ball model)، مما يسمح للشبكة بالتركيز
    على الميزات الأكثر صلة بطريقة ت تراعي البنية الكامنة للبيانات.

3.  **مُحسِّن الشواش (Chaos Optimizer):** (اختياري، يتم التحكم به عبر flag)
    مُحسِّن تجريبي مستوحى من نظرية الشواش (Chaos Theory)، وتحديدًا نظام لورنز.
    يهدف إلى مساعدة الشبكة على الهروب من النقاط الصغرى المحلية (local minima)
    أثناء التدريب واستكشاف فضاء المعلمات بشكل أفضل من خلال إدخال ديناميكيات
    غير خطية وشواشية في عملية تحديث الأوزان. (مُعطّل افتراضيًا لصالح Adam من أجل الاستقرار).

4.  **مقياس Tau للتعلم الموجه بالمخاطر:**
    بدلاً من تعظيم المكافأة المتراكمة فقط، يتعلم الوكيل باستخدام مقياس "Tau"
    المبتكر. Tau يوازن بين التقدم المحرز نحو الهدف (المكافأة الإيجابية)
    والمخاطرة المتوقعة للوصول إلى ذلك التقدم (الناتجة من رأس مخصص في الشبكة).
    دالة الخسارة مصممة لتقليل المخاطرة المتوقعة في الحالات التي أدت تاريخيًا
    إلى قيم Tau عالية، مما يشجع الوكيل على إيجاد مسارات فعالة وآمنة.

5.  **تنظيم الإنتروبيا (Entropy Regularization):**
    تقنية قياسية في التعلم المعزز تم إضافتها هنا لتعزيز الاستكشاف ومنع السياسة
    من أن تصبح حتمية بشكل مفرط، مما يساعد على تحسين الاستقرار والاتساق في الأداء.

الاستخدام:
    - قم بتشغيل السكربت مباشرة.
    - يمكن تعديل الإعدادات الرئيسية (مثل حجم الشبكة، معدل التعلم، استخدام
      محسن الشواش، معامل الإنتروبيا) في قسم `if __name__ == "__main__":`.
    - سيقوم السكربت بتدريب الوكيل ثم اختباره وعرض النتائج ورسم بياني للأداء.

الاعتماديات:
    - Python 3.x
    - PyTorch
    - NumPy
    - Matplotlib (اختياري، للرسم البياني)
    - Pandas (اختياري، للرسم البياني - المتوسط المتحرك)
"""

# ========================================
# 0. الاستيرادات الضرورية
# ========================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math # لاستخدام acosh في الانتباه الزائدي
import sys # للتحقق من المشاكل المحتملة
import torch.distributions # Needed for entropy calculation

# ========================================
# 1. البيئة: GridWorld المتقدمة
# ========================================
class AdvancedGridWorld:
    """
    يمثل بيئة عالم شبكي (GridWorld) ثنائية الأبعاد مع:
    - حجم قابل للتحديد.
    - موقع هدف ثابت في الزاوية السفلية اليمنى.
    - مجموعة من العقبات الثابتة.
    - وكيل يبدأ من الزاوية العلوية اليسرى (أو موقع آمن بديل).
    - أربعة إجراءات ممكنة: أعلى، يمين، أسفل، يسار.
    - نظام مكافآت: +10 للوصول للهدف، -5 للاصطدام بعقبة، -0.1 لكل خطوة.
    """
    def __init__(self, size=5):
        """
        تهيئة البيئة.

        Args:
            size (int): حجم الجانب الواحد من الشبكة المربعة.
        """
        self.size = size
        # تحديد موقع الهدف
        self.goal = (size - 1, size - 1)
        # تحديد مواقع العقبات
        self.obstacles = [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)]
        # إزالة الهدف إذا كان بالصدفة ضمن قائمة العقبات
        if self.goal in self.obstacles:
            self.obstacles.remove(self.goal)
        # End if statement

        # تحديد موقع بداية آمن (ليس عقبة وليس هدفًا)
        start_pos = (0, 0)
        while start_pos in self.obstacles or start_pos == self.goal:
            start_pos = (random.randint(0, size-1), random.randint(0, size-1))
        # End while loop
        self.agent_pos = start_pos
        self.start_pos = start_pos # الاحتفاظ بموقع البداية لإعادة التعيين
        # تحديد عدد الإجراءات الممكنة
        self.action_space_n = 4
    # End __init__ method

    def reset(self):
        """
        إعادة تعيين البيئة إلى حالتها الأولية.

        Returns:
            np.ndarray: الحالة الأولية للشبكة كمتجه مسطح.
        """
        self.agent_pos = self.start_pos
        state = self._get_state()
        return state
    # End reset method

    def _get_state(self):
        """
        يحسب تمثيل الحالة الحالية للبيئة.

        Returns:
            np.ndarray: متجه مسطح يمثل الشبكة.
                        1.0: موقع العميل
                        0.5: موقع الهدف
                       -1.0: موقع العقبة
                        0.0: خانة فارغة
        """
        # إنشاء شبكة فارغة
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        # وضع العميل
        agent_r, agent_c = self.agent_pos
        if 0 <= agent_r < self.size and 0 <= agent_c < self.size:
             grid[self.agent_pos] = 1.0
        # End if statement
        # وضع الهدف
        goal_r, goal_c = self.goal
        if 0 <= goal_r < self.size and 0 <= goal_c < self.size:
             grid[self.goal] = 0.5
        # End if statement
        # وضع العقبات
        for obs_x, obs_y in self.obstacles:
            if 0 <= obs_x < self.size and 0 <= obs_y < self.size:
                grid[obs_x, obs_y] = -1.0
            # End if statement
        # End for loop
        # تسوية الشبكة إلى متجه
        flat_grid = grid.flatten()
        return flat_grid
    # End _get_state method

    def step(self, action):
        """
        تنفيذ إجراء في البيئة وتحديث حالتها.

        Args:
            action (int): الإجراء المراد تنفيذه (0: أعلى, 1: يمين, 2: أسفل, 3: يسار).

        Returns:
            tuple: يحتوي على (next_state, reward, done, info)
                   next_state (np.ndarray): الحالة الجديدة للبيئة.
                   reward (float): المكافأة المستلمة بعد تنفيذ الإجراء.
                   done (bool): هل انتهت الحلقة (وصل للهدف أو شرط آخر).
                   info (dict): معلومات إضافية (فارغة حاليًا).
        """
        current_x, current_y = self.agent_pos
        next_x = current_x
        next_y = current_y

        # تحديث الموقع المحتمل بناءً على الإجراء
        if action == 0:
            next_x = max(0, current_x - 1)
        elif action == 1:
            next_y = min(self.size - 1, current_y + 1)
        elif action == 2:
            next_x = min(self.size - 1, current_x + 1)
        elif action == 3:
            next_y = max(0, current_y - 1)
        else:
            # التعامل مع إجراء غير صالح
            print(f"Warning: Invalid action {action} received!")
        # End if/elif/else block

        new_pos = (next_x, next_y)
        # تكلفة الحركة الافتراضية
        reward = -0.1
        done = False

        # التحقق من النتائج
        if new_pos == self.goal:
            # الوصول للهدف
            reward = 10.0
            done = True
            # تحديث الموقع
            self.agent_pos = new_pos
        elif new_pos in self.obstacles:
            # الاصطدام بعقبة
            reward = -5.0
            # البقاء في المكان الحالي (لا يتم تحديث الموقع)
            new_pos = self.agent_pos
            # يمكن جعل الحلقة تنتهي هنا أيضًا إذا أردنا
            # done = True
        else:
             # حركة صالحة إلى خانة فارغة
             # تحديث الموقع
             self.agent_pos = new_pos
        # End if/elif/else block for rewards/position update

        # الحصول على الحالة الجديدة بعد التحديث
        next_state = self._get_state()
        # معلومات إضافية (غير مستخدمة حاليًا)
        info = {}
        return next_state, reward, done, info
    # End step method

    def render(self):
        """
        عرض الحالة الحالية للبيئة كنص في الطرفية.
        'A': العميل
        'G': الهدف
        'X': عقبة
        '.': خانة فارغة
        """
        # إنشاء شبكة نصية فارغة
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        # وضع الهدف
        goal_r, goal_c = self.goal
        if 0 <= goal_r < self.size and 0 <= goal_c < self.size:
             grid[self.goal[0]][self.goal[1]] = 'G'
        # End if statement
        # وضع العقبات
        for obs_x, obs_y in self.obstacles:
            if 0 <= obs_x < self.size and 0 <= obs_y < self.size:
                grid[obs_x][obs_y] = 'X'
            # End if statement
        # End for loop
        # وضع العميل
        agent_r, agent_c = self.agent_pos
        if 0 <= agent_r < self.size and 0 <= agent_c < self.size:
             grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        else:
            # تحذير إذا كان العميل خارج الحدود (لا ينبغي أن يحدث)
            print(f"Warning: Agent position {self.agent_pos} out of bounds during render!")
        # End if/else statement

        # طباعة الشبكة مع حدود
        print("-" * (self.size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        # End for loop
        print("-" * (self.size * 2 + 1))
        # طباعة معلومات إضافية
        print(f"Agent @ {self.agent_pos}, Goal @ {self.goal}")
        print() # سطر فارغ للمسافة
    # End render method
# End AdvancedGridWorld class


# ========================================
# 2. بناء المكونات المبتكرة
# ========================================

# ----------------------------------------
# 2.أ. طبقة الارتباط الديناميكي
# ----------------------------------------
class DynamicCorrelationLayer(nn.Module):
    """
    [ابتكار 1] طبقة تهدف لاكتشاف الارتباطات الديناميكية بين ميزات الإدخال.
    تستخدم مصفوفة ارتباط (covariance-like) قابلة للتعلم، متبوعة بتعديل طور
    غير خطي (phase modulation) واتصال تخطي (residual connection).
    """
    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): بُعد متجه الميزات المدخلة (والمخرجة).
        """
        super().__init__()
        # مصفوفة الارتباط القابلة للتعلم
        self.cov_matrix = nn.Parameter(torch.randn(input_dim, input_dim))
        # طبقة تعديل الطور (غير خطية)
        self.phase_mod = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh() # دالة Tanh للحفاظ على نطاق محدد للمخرجات
        )
    # End __init__ method

    def forward(self, x):
        """
        تمرير البيانات عبر الطبقة.

        Args:
            x (torch.Tensor): متجه ميزات الإدخال (قد يكون دفعة أو فردي).

        Returns:
            torch.Tensor: متجه الميزات بعد تطبيق الارتباط وتعديل الطور.
        """
        # التعامل مع الدفعات
        is_batched = x.dim() > 1
        if not is_batched:
            # إضافة بُعد الدفعة إذا كان الإدخال فرديًا
            x = x.unsqueeze(0)
        # End if statement

        # 1. تطبيق مصفوفة الارتباط
        correlated = torch.matmul(x, self.cov_matrix)
        # 2. تطبيق تعديل الطور غير الخطي
        phased = self.phase_mod(correlated)
        # 3. إضافة اتصال تخطي (يساعد في استقرار التدريب)
        output = phased + x

        # إزالة بُعد الدفعة إذا تمت إضافته
        if not is_batched:
            output = output.squeeze(0)
        # End if statement
        return output
    # End forward method
# End DynamicCorrelationLayer class

# ----------------------------------------
# 2.ب. الانتباه الزائدي
# ----------------------------------------
class HyperbolicAttention(nn.Module):
    """
    [ابتكار 2] آلية انتباه تستخدم الهندسة الزائدية لحساب أوزان الانتباه.
    تفترض أن العلاقات بين الميزات (خاصة المكانية أو الهرمية) يمكن تمثيلها
    بشكل أفضل في فضاء زائدي. تستخدم نموذج كرة بوانكاريه (Poincaré Ball) لحساب
    المسافات.
    """
    def __init__(self, dim):
        """
        Args:
            dim (int): بُعد متجهات الميزات (query, key, value).
        """
        super().__init__()
        # معلمة انحناء الفضاء الزائدي (قابلة للتعلم)
        self.c = nn.Parameter(torch.tensor(0.1))
        # ثابت صغير لتحسين الاستقرار العددي (تجنب القسمة على صفر أو log(0))
        self.epsilon = 1e-6
    # End __init__ method

    def _poincare_distance(self, u, v):
        """
        يحسب المسافة الزائدية بين متجهين u و v في نموذج كرة بوانكاريه.

        Args:
            u (torch.Tensor): المتجه الأول (أو دفعة من المتجهات).
            v (torch.Tensor): المتجه الثاني (أو دفعة من المتجهات).

        Returns:
            torch.Tensor: المسافة (أو المسافات) الزائدية.
        """
        # تقييد قيمة الانحناء c لتكون موجبة أثناء الحساب لضمان الاستقرار
        clamp_c = torch.clamp(self.c, min=self.epsilon)
        # حساب الجذر التربيعي لـ c (مع التأكد من عدم السلبية)
        c_sqrt = torch.sqrt(clamp_c)

        # حساب مربعات معايير L2 للمتجهات وفروقها
        uu_norm_sq = torch.sum(u * u, dim=-1, keepdim=True)
        vv_norm_sq = torch.sum(v * v, dim=-1, keepdim=True)
        uv_diff_norm_sq = torch.sum((u - v) * (u - v), dim=-1, keepdim=True)

        # حساب المقام في صيغة المسافة مع تقييد لتجنب القيم الصغيرة جدًا
        term1 = 1.0 - clamp_c * uu_norm_sq
        term2 = 1.0 - clamp_c * vv_norm_sq
        clamped_term1 = torch.clamp(term1, min=self.epsilon)
        clamped_term2 = torch.clamp(term2, min=self.epsilon)
        denominator = clamped_term1 * clamped_term2

        # حساب الوسيط لدالة acosh (يجب أن يكون >= 1)
        numerator = 2.0 * clamp_c * uv_diff_norm_sq
        argument = 1.0 + numerator / denominator
        # تقييد الوسيط لضمان أنه >= 1 + epsilon
        argument = torch.clamp(argument, min=1.0 + self.epsilon)

        # إضافة epsilon إضافية داخل acosh لزيادة استقرار حساب التدرج
        stable_argument = argument + self.epsilon

        # حساب المسافة الزائدية النهائية
        distance = (1.0 / c_sqrt) * torch.acosh(stable_argument)

        # التحقق من وجود قيم NaN (غير رقمية) ومعالجتها إذا لزم الأمر
        nan_mask = torch.isnan(distance)
        if torch.any(nan_mask):
            # طباعة تحذير واستبدال NaN بقيمة كبيرة (تشير إلى مسافة كبيرة)
            print("WARNING: NaN detected in hyperbolic distance calculation! Replacing with 10.0")
            distance = torch.where(nan_mask, torch.full_like(distance, 10.0), distance)
        # End if statement

        return distance
    # End _poincare_distance method

    def forward(self, query, keys, values):
        """
        حساب مخرجات الانتباه الزائدي.

        Args:
            query (torch.Tensor): متجه الاستعلام (أو دفعة).
            keys (torch.Tensor): متجهات المفاتيح (أو دفعة).
            values (torch.Tensor): متجهات القيم (أو دفعة).
                                  (في حالة الانتباه الذاتي، تكون query=keys=values).

        Returns:
            torch.Tensor: المخرجات الموزونة بعد تطبيق الانتباه الزائدي.
        """
        # التعامل مع الدفعات للاستعلام
        is_batched_query = query.dim() > 1
        if not is_batched_query:
            query = query.unsqueeze(0)
        # End if statement

        # تنفيذ الانتباه الذاتي (الاستخدام الحالي في هذا الكود)
        if keys is query and values is query:
            batch_size, dim = query.shape
            # توسيع الأبعاد لحساب المسافات بين كل زوج في الدفعة
            query_expanded = query.unsqueeze(1)
            query_expanded = query_expanded.expand(batch_size, batch_size, dim)
            keys_expanded = keys.unsqueeze(0)
            keys_expanded = keys_expanded.expand(batch_size, batch_size, dim)
            values_expanded = values.unsqueeze(0)
            values_expanded = values_expanded.expand(batch_size, batch_size, dim)

            # حساب المسافات الزائدية وإزالة البعد الإضافي
            distances = self._poincare_distance(query_expanded, keys_expanded)
            distances = distances.squeeze(-1)

            # حساب أوزان الانتباه باستخدام softmax على (-المسافة)
            # (المسافة الأقل تعني وزنًا أعلى)
            attn_weights = torch.softmax(-distances, dim=-1)

            # التحقق من وجود NaN في الأوزان ومعالجتها
            nan_mask_weights = torch.isnan(attn_weights)
            if torch.any(nan_mask_weights):
                print("WARNING: NaN detected in attention weights! Replacing with zeros and renormalizing.")
                attn_weights = torch.where(nan_mask_weights, torch.zeros_like(attn_weights), attn_weights)
                row_sums = attn_weights.sum(dim=-1, keepdim=True)
                # إعادة تطبيع الأوزان لضمان مجموعها = 1
                safe_row_sums = row_sums + self.epsilon
                attn_weights = attn_weights / safe_row_sums
            # End if statement

            # حساب المخرجات الموزونة باستخدام ضرب المصفوفات المجمعة
            attn_weights_unsqueezed = attn_weights.unsqueeze(1)
            output = torch.bmm(attn_weights_unsqueezed, values_expanded)
            # إزالة البعد الأوسط
            output = output.squeeze(1)
        else:
             # حالة الانتباه غير الذاتي (غير مطبقة بالكامل هنا)
             raise NotImplementedError("Non-self attention case not fully implemented/verified.")
        # End if/else block

        # إزالة بُعد الدفعة إذا تمت إضافته في البداية
        if not is_batched_query:
            output = output.squeeze(0)
        # End if statement
        return output
    # End forward method
# End HyperbolicAttention class

# ----------------------------------------
# 2.ج. محسن الشواش (اختياري)
# ----------------------------------------
class ChaosOptimizer(optim.Optimizer):
    """
    [ابتكار 3] مُحسِّن تجريبي مستوحى من نظام لورنز الشواشي.
    يهدف إلى تحسين استكشاف فضاء المعلمات والهروب من النقاط الصغرى المحلية.
    *ملاحظة: هذا المحسن تجريبي وقد يكون أقل استقرارًا من المحسنات القياسية.*
    """
    def __init__(self, params, lr=1e-3, sigma=10.0, rho=28.0, beta=8.0/3.0, weight_decay=0):
        """
        Args:
            params: معلمات النموذج المراد تحسينها.
            lr (float): معدل التعلم.
            sigma, rho, beta (float): معلمات نظام لورنز.
            weight_decay (float): معامل تنظيم L2 (تخفيف الوزن).
        """
        # التحقق من صحة المعلمات الفائقة
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # End if statement
        if sigma < 0.0:
            raise ValueError(f"Invalid sigma value: {sigma}")
        # End if statement
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        # End if statement

        # إعداد القيم الافتراضية
        defaults = dict(lr=lr, sigma=sigma, rho=rho, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)
    # End __init__ method

    @torch.no_grad() # لا نحتاج لحساب تدرجات لخطوة التحديث نفسها
    def step(self, closure=None):
        """
        تنفيذ خطوة تحسين واحدة.

        Args:
            closure (callable, optional): دالة closure تعيد حساب الخسارة (غير مستخدمة غالبًا هنا).

        Returns:
            float or None: قيمة الخسارة إذا تم حسابها بواسطة closure.
        """
        loss = None
        if closure is not None:
            # تمكين حساب التدرجات إذا كان closure يتطلب ذلك
            with torch.enable_grad():
                loss = closure()
            # End with block
        # End if statement

        # المرور على مجموعات المعلمات المختلفة (قد يكون لكل منها إعدادات مختلفة)
        for group in self.param_groups:
            # استخلاص المعلمات الفائقة لهذه المجموعة
            lr = group['lr']
            sigma = group['sigma']
            rho = group['rho']
            beta = group['beta']
            weight_decay = group['weight_decay']

            # المرور على كل معلمة في المجموعة
            for p in group['params']:
                # تخطي المعلمة إذا لم يكن لها تدرج (gradient)
                if p.grad is None:
                    continue
                # End if statement
                grad = p.grad

                # التحقق من وجود NaN أو Inf في التدرجات
                grad_is_nan = torch.isnan(grad)
                grad_is_inf = torch.isinf(grad)
                if torch.any(grad_is_nan) or torch.any(grad_is_inf):
                    print(f"WARNING: NaN or Inf detected in gradients for parameter {list(p.shape)}! Skipping update for this parameter.")
                    continue # تخطي تحديث هذه المعلمة
                # End if statement

                # التحقق من وجود NaN أو Inf في بيانات المعلمة نفسها (حالة خطيرة)
                param_data = p.data
                param_is_nan = torch.isnan(param_data)
                param_is_inf = torch.isinf(param_data)
                if torch.any(param_is_nan) or torch.any(param_is_inf):
                     print(f"ERROR: NaN or Inf detected in parameter data {list(p.shape)}! Training cannot continue reliably.")
                     raise ValueError(f"NaN/Inf in parameter data {list(p.shape)}") # إيقاف التدريب
                 # End if statement

                # تطبيق تخفيف الوزن L2 إذا تم تحديده
                decayed_grad = grad
                if weight_decay != 0:
                    decayed_grad = grad.add(param_data, alpha=weight_decay)
                # End if statement

                # تطبيق معادلات لورنز المعدلة باستخدام التدرج وبيانات المعلمة
                dx = sigma * (decayed_grad - param_data)
                dy = param_data * (rho - decayed_grad) - decayed_grad
                dz = decayed_grad * param_data - beta * decayed_grad
                # حساب التحديث الشواشي الكلي
                chaotic_update = dx + dy + dz

                # التحقق من وجود NaN أو Inf في قيمة التحديث النهائية
                update_is_nan = torch.isnan(chaotic_update)
                update_is_inf = torch.isinf(chaotic_update)
                if torch.any(update_is_nan) or torch.any(update_is_inf):
                    print(f"WARNING: NaN or Inf detected in chaotic update for parameter {list(p.shape)}! Skipping update.")
                    continue # تخطي التحديث
                # End if statement

                # تحديث المعلمة باستخدام التحديث الشواشي ومعدل التعلم
                p.data.add_(chaotic_update, alpha=lr)
            # End inner for loop (parameters)
        # End outer for loop (param_groups)
        return loss
    # End step method
# End ChaosOptimizer class


# ========================================
# 3. الشبكة العصبية المبتكرة (السياسة)
# ========================================
class TauPolicyNetwork(nn.Module):
    """
    الشبكة العصبية للسياسة (Policy Network) التي تدمج المكونات المبتكرة.
    تتكون من:
    - طبقة تشفير أولية (Encoder).
    - طبقة الارتباط الديناميكي (DynamicCorrelationLayer).
    - طبقة الانتباه الزائدي (HyperbolicAttention).
    - رأس الممثل (Actor Head) لتحديد احتمالات الإجراءات.
    - رأس مقدر المخاطر (Risk Predictor Head) لتقدير المخاطرة المتوقعة.
    """
    def __init__(self, input_dim, hidden_dim, action_dim):
        """
        Args:
            input_dim (int): بُعد متجه الحالة المدخلة.
            hidden_dim (int): بُعد الطبقات المخفية.
            action_dim (int): عدد الإجراءات الممكنة (بُعد المخرجات للممثل).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 1. طبقة التشفير الأولية (يمكن جعلها أعمق)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() # استخدام ReLU كدالة تنشيط شائعة
        )
        # 2. طبقة الارتباط الديناميكي
        self.correlator = DynamicCorrelationLayer(hidden_dim)
        # 3. طبقة الانتباه الزائدي (تُطبق كانتباه ذاتي هنا)
        self.attention = HyperbolicAttention(hidden_dim)
        # 4. رأس الممثل (يخرج لوغاريتمات الاحتمالات - logits)
        self.actor = nn.Linear(hidden_dim, action_dim)
        # 5. رأس مقدر المخاطر (يخرج لوغاريتم المخاطرة قبل sigmoid)
        self.risk_predictor = nn.Linear(hidden_dim, 1)

        # ثابت صغير لمنع القسمة على صفر أو log(0)
        self.epsilon = 1e-8

        # تهيئة انحياز مقدر المخاطر ليبدأ بتوقع مخاطر منخفضة
        # هذا يساعد على منع الانهيار الفوري نحو توقع مخاطرة عالية أو منخفضة جدًا
        with torch.no_grad():
            # قيمة -2.0 تجعل ناتج sigmoid حوالي 0.12
            self.risk_predictor.bias.fill_(-2.0)
        # End with block
    # End __init__ method

    def forward(self, state):
        """
        تمرير الحالة عبر الشبكة للحصول على مخرجات السياسة والمخاطرة.

        Args:
            state (torch.Tensor or np.ndarray): متجه الحالة المدخلة (دفعة أو فردي).

        Returns:
            tuple: يحتوي على (action_logits, action_probs, risk)
                   action_logits (torch.Tensor): لوغاريتمات الاحتمالات لكل إجراء.
                   action_probs (torch.Tensor): الاحتمالات المحسوبة لكل إجراء (بعد softmax).
                   risk (torch.Tensor): المخاطرة المتوقعة (قيمة بين 0 و 1 بعد sigmoid).
        """
        # تحويل الإدخال إلى Tensor إذا لزم الأمر
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        # End if statement
        # التعامل مع الدفعات
        is_batched = state.dim() > 1
        if not is_batched:
            state = state.unsqueeze(0)
        # End if statement
        state = state.float() # التأكد من أن النوع هو float

        # تمرير البيانات عبر الطبقات بالتسلسل
        encoded_state = self.encoder(state)
        correlated_state = self.correlator(encoded_state)

        # التحقق من NaN/Inf قبل الانتباه
        corr_state_is_nan = torch.isnan(correlated_state)
        corr_state_is_inf = torch.isinf(correlated_state)
        if torch.any(corr_state_is_nan) or torch.any(corr_state_is_inf):
             print(f"!!! WARNING: NaN/Inf detected in correlated_state before attention! Applying nan_to_num.")
             correlated_state = torch.nan_to_num(correlated_state, nan=0.0, posinf=1.0, neginf=-1.0)
        # End if statement

        attn_output = self.attention(correlated_state, correlated_state, correlated_state)

        # التحقق من NaN/Inf بعد الانتباه
        attn_is_nan = torch.isnan(attn_output)
        attn_is_inf = torch.isinf(attn_output)
        if torch.any(attn_is_nan) or torch.any(attn_is_inf):
             print(f"!!! WARNING: NaN/Inf detected in attn_output after attention! Applying nan_to_num.")
             attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1.0, neginf=-1.0)
        # End if statement

        # حساب مخرجات رؤوس الشبكة
        action_logits = self.actor(attn_output)
        risk_logits = self.risk_predictor(attn_output)

        # التحقق من NaN/Inf في اللوغاريتمات
        logits_are_nan = torch.isnan(action_logits)
        logits_are_inf = torch.isinf(action_logits)
        if torch.any(logits_are_nan) or torch.any(logits_are_inf):
             print(f"!!! WARNING: NaN/Inf detected in action_logits! Applying nan_to_num.")
             action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        # End if statement

        # حساب الاحتمالات والمخاطرة النهائية
        action_probs = torch.softmax(action_logits, dim=-1)
        risk = torch.sigmoid(risk_logits)

        # التحقق النهائي من صلاحية الاحتمالات
        probs_are_nan = torch.isnan(action_probs)
        probs_are_inf = torch.isinf(action_probs)
        probs_are_neg = action_probs < 0
        probs_sum = action_probs.sum(dim=-1)
        sum_is_close_to_one_tensor = torch.allclose(probs_sum, torch.tensor(1.0, device=action_probs.device), atol=1e-5)
        if torch.any(probs_are_nan) or torch.any(probs_are_inf) or torch.any(probs_are_neg) or not sum_is_close_to_one_tensor:
            print(f"!!! WARNING: Invalid action_probs calculated! Using uniform distribution as fallback.")
            action_probs = torch.ones_like(action_probs)
            action_probs = action_probs / action_probs.shape[-1]
            # ضبط اللوغاريتمات لتكون متسقة (صفر يعني توزيع موحد)
            action_logits = torch.zeros_like(action_logits)
        # End if statement

        # التحقق النهائي من صلاحية المخاطرة
        risk_is_nan = torch.isnan(risk)
        risk_is_inf = torch.isinf(risk)
        if torch.any(risk_is_nan) or torch.any(risk_is_inf):
             print(f"!!! WARNING: NaN/Inf detected in risk output! Clamping to [eps, 1-eps].")
             risk = torch.nan_to_num(risk, nan=0.5, posinf=1.0, neginf=0.0)
             risk = torch.clamp(risk, min=self.epsilon, max=1.0-self.epsilon)
         # End if statement

        # إزالة بُعد الدفعة إذا لم يكن موجودًا في الأصل
        if not is_batched:
            action_logits = action_logits.squeeze(0)
            action_probs = action_probs.squeeze(0)
            risk = risk.squeeze(0)
            risk = risk.squeeze(-1)
        # End if statement

        # إرجاع اللوغاريتمات، الاحتمالات، والمخاطرة
        return action_logits, action_probs, risk
    # End forward method
# End TauPolicyNetwork class

# ========================================
# 4. خوارزمية التدريب (الوكيل)
# ========================================
class TauAgent:
    """
    الوكيل الذي يتعلم السياسة باستخدام شبكة TauPolicyNetwork والمكونات المبتكرة.
    يستخدم ذاكرة إعادة التشغيل (Replay Memory) وخوارزمية تحديث تعتمد على مقياس Tau
    وتنظيم الإنتروبيا.
    """
    def __init__(self, env, hidden_dim=128, learning_rate=1e-4, gamma=0.99, memory_size=10000, batch_size=64, clip_grad_norm=1.0, use_chaos_optimizer=False, weight_decay=0, entropy_coeff=0.01):
        """
        Args:
            env (AdvancedGridWorld): بيئة التدريب.
            hidden_dim (int): حجم الطبقات المخفية في الشبكة.
            learning_rate (float): معدل التعلم للمُحسِّن.
            gamma (float): معامل الخصم (غير مستخدم مباشرة في الخسارة ولكن جيد للاحتفاظ به).
            memory_size (int): حجم ذاكرة إعادة التشغيل.
            batch_size (int): حجم الدفعة المستخدمة في التحديث.
            clip_grad_norm (float or None): القيمة القصوى لمعيار التدرج (للتقليم).
            use_chaos_optimizer (bool): هل يتم استخدام محسن الشواش أم Adam.
            weight_decay (float): معامل تخفيف الوزن L2.
            entropy_coeff (float): معامل تنظيم الإنتروبيا.
        """
        self.env = env
        self.input_dim = env.size * env.size
        self.action_dim = env.action_space_n
        self.hidden_dim = hidden_dim
        # إنشاء شبكة السياسة
        self.policy_net = TauPolicyNetwork(self.input_dim, self.hidden_dim, self.action_dim)
        # إنشاء ذاكرة إعادة التشغيل
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma # معامل الخصم (غير مستخدم حاليًا بشكل مباشر)
        self.tau_epsilon = 1e-8 # ثابت صغير لحساب Tau
        self.clip_grad_norm = clip_grad_norm # قيمة تقليم التدرج
        self.entropy_coeff = entropy_coeff # معامل الإنتروبيا

        # اختيار المُحسِّن بناءً على العلامة
        if use_chaos_optimizer:
            self.optimizer_type = "Chaos"
        else:
            self.optimizer_type = "Adam"
        # End if/else
        print(f"--- Using {self.optimizer_type} Optimizer ---")
        if use_chaos_optimizer:
            # استخدام معدل تعلم منخفض جدًا لمحسن الشواش كإجراء وقائي
            effective_lr = min(learning_rate, 1e-5)
            print(f"Using Chaos Optimizer with LR: {effective_lr}")
            self.optimizer = ChaosOptimizer(self.policy_net.parameters(), lr=effective_lr, weight_decay=weight_decay)
        else:
            # استخدام Adam مع معدل التعلم المحدد
            print(f"Using Adam Optimizer with LR: {learning_rate}")
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # End if/else block choosing optimizer
    # End __init__ method


    def select_action(self, state):
        """
        اختيار إجراء بناءً على الحالة الحالية باستخدام سياسة الشبكة.
        يتم أخذ عينة من توزيع الاحتمالات لتضمين بعض الاستكشاف.

        Args:
            state (np.ndarray): متجه الحالة الحالي.

        Returns:
            tuple: يحتوي على (action_item, risk_item)
                   action_item (int): الإجراء المختار.
                   risk_item (float): المخاطرة المتوقعة من الشبكة لهذا الإجراء.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # الحصول على المخرجات من الشبكة بدون حساب تدرجات
        with torch.no_grad():
            # نحتاج فقط للاحتمالات والمخاطرة هنا
            action_logits, probs, risk = self.policy_net(state_tensor)
        # End with block

        # التحقق من صلاحية الاحتمالات قبل أخذ العينة
        probs_are_finite = torch.isfinite(probs)
        probs_are_non_neg = probs >= 0
        if not torch.all(probs_are_finite) or not torch.all(probs_are_non_neg):
            probs_np = probs.detach().cpu().numpy()
            print(f"!!! ERROR: Invalid probabilities before sampling: {probs_np}. Choosing random action.")
            # اختيار إجراء عشوائي كحل بديل
            action_item = random.randrange(self.action_dim)
            risk_is_finite = torch.isfinite(risk)
            risk_item = risk.item() if risk_is_finite else 0.5
        else:
            # إعادة تطبيع بسيطة إذا كان المجموع ليس 1 تمامًا
            probs_sum = probs.sum()
            sum_is_close_to_one = torch.allclose(probs_sum, torch.tensor(1.0, device=probs.device), atol=1e-5)
            if not sum_is_close_to_one:
                safe_probs_sum = probs_sum + self.tau_epsilon
                probs = probs / safe_probs_sum
            # End if statement

            # أخذ عينة من التوزيع الاحتمالي
            try:
                action_dist = torch.distributions.Categorical(probs=probs)
                action = action_dist.sample()
                action_item = action.item()
                risk_item = risk.item()
            except RuntimeError as e:
                # التعامل مع أخطاء محتملة في أخذ العينات
                print(f"!!! ERROR during sampling: {e}")
                probs_np = probs.detach().cpu().numpy()
                print(f"Probabilities causing error: {probs_np}")
                # اختيار إجراء عشوائي كحل بديل
                action_item = random.randrange(self.action_dim)
                risk_is_finite = torch.isfinite(risk)
                risk_item = risk.item() if risk_is_finite else 0.5
            # End try/except block
        # End if/else block checking prob validity

        return action_item, risk_item
    # End select_action method

    def store_experience(self, state, action, reward, next_state, done, risk):
        """
        [ابتكار 4] حساب قيمة Tau وتخزين التجربة في الذاكرة.
        Tau = (التقدم + ثابت صغير) / (المخاطرة المتوقعة + ثابت صغير آخر + epsilon)

        Args:
            state (np.ndarray): الحالة قبل الإجراء.
            action (int): الإجراء المتخذ.
            reward (float): المكافأة المستلمة.
            next_state (np.ndarray): الحالة بعد الإجراء.
            done (bool): هل انتهت الحلقة.
            risk (float): المخاطرة التي توقعتها الشبكة عند اختيار الإجراء.
        """
        # اعتبار التقدم فقط كمكافأة إيجابية
        progress = max(0.0, reward)
        # تقييد المخاطرة المتوقعة ضمن نطاق آمن قبل الحساب
        safe_risk = np.clip(risk, self.tau_epsilon, 1.0 - self.tau_epsilon)
        # استخدام ثوابت صغيرة (0.1) لتعديل السلوك ومنع القيم المتطرفة لـ Tau
        denominator = safe_risk + 0.1 + self.tau_epsilon
        tau = (progress + 0.1) / denominator
        # تخزين التجربة مع قيمة Tau المحسوبة
        experience = (state, action, tau, next_state, done)
        self.memory.append(experience)
    # End store_experience method

    def _update_policy(self):
        """
        تحديث سياسة الشبكة باستخدام دفعة عشوائية من التجارب المخزنة.
        يحسب الخسارة المجمعة (خسارة المخاطرة + تنظيم الإنتروبيا) وينفذ خطوة تحسين.
        """
        # التحقق من وجود عدد كافٍ من العينات للتدريب
        if len(self.memory) < self.batch_size:
            return None # لا تقم بالتحديث
        # End if statement

        # أخذ عينة عشوائية من الذاكرة
        batch = random.sample(self.memory, self.batch_size)
        # فك حزم العينات
        states = [exp[0] for exp in batch]
        actions = [exp[1] for exp in batch]
        taus = [exp[2] for exp in batch]
        next_states = [exp[3] for exp in batch] # غير مستخدمة حاليًا في الخسارة
        dones = [exp[4] for exp in batch] # غير مستخدمة حاليًا في الخسارة

        # تحويل البيانات إلى تنسورات
        state_batch = torch.tensor(np.array(states), dtype=torch.float32)
        tau_batch = torch.tensor(taus, dtype=torch.float32)
        tau_batch = tau_batch.unsqueeze(1) # إضافة بُعد الميزة

        # الحصول على المخرجات الحالية للشبكة (نحتاج اللوغاريتمات والمخاطر)
        action_logits_batch, current_probs, current_risks = self.policy_net(state_batch)

        # التحقق من صلاحية المخاطر قبل استخدامها في الخسارة
        risks_are_finite = torch.isfinite(current_risks)
        risks_in_range = (current_risks >= 0) & (current_risks <= 1)
        if not torch.all(risks_are_finite) or not torch.all(risks_in_range):
             print("!!! WARNING: Invalid risks detected before loss calculation. Clamping risks.")
             current_risks = torch.clamp(current_risks, min=self.tau_epsilon, max=1.0-self.tau_epsilon)
        # End if statement

        # --- حساب الخسارة 1: خسارة المخاطرة الموجهة بـ Tau ---
        # [ابتكار 4 مطبق هنا]
        # استخدام الطريقة التي تعمل حاليًا: risk * tau
        safe_current_risks = torch.clamp(current_risks, min=self.tau_epsilon, max=1.0 - self.tau_epsilon)
        risk_loss = (safe_current_risks * tau_batch).mean()

        # --- حساب الخسارة 2: تنظيم الإنتروبيا ---
        # [ابتكار 5 مطبق هنا]
        # استخدام اللوغاريتمات لحساب توزيع احتمالي مستقر
        policy_dist = torch.distributions.Categorical(logits=action_logits_batch)
        # حساب متوسط الإنتروبيا على الدفعة
        entropy_bonus = policy_dist.entropy().mean()

        # --- الخسارة المجمعة ---
        # الهدف: تقليل (risk * tau) وزيادة (entropy_bonus)
        combined_loss = risk_loss - self.entropy_coeff * entropy_bonus

        # التحقق من صلاحية الخسارة النهائية
        loss_is_finite = torch.isfinite(combined_loss)
        if not loss_is_finite:
            loss_item = combined_loss.item()
            print(f"!!! ERROR: Combined Loss is NaN or Inf ({loss_item}). Skipping backpropagation.")
            # طباعة مكونات الخسارة للمساعدة في التشخيص
            risk_loss_item = risk_loss.item()
            entropy_item = entropy_bonus.item()
            print(f"Contributing Factors - Risk Loss: {risk_loss_item}, Entropy Bonus: {entropy_item}")
            return None # تخطي التحديث
        # End if statement

        # --- تنفيذ خطوة التحسين ---
        # 1. تصفير التدرجات السابقة
        self.optimizer.zero_grad()
        # 2. حساب التدرجات الجديدة (الانتشار الخلفي)
        combined_loss.backward()

        # 3. تقليم التدرج (اختياري ولكن موصى به لمنع الانفجار)
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.clip_grad_norm)
        # End if statement

        # 4. تحديث الأوزان باستخدام المُحسِّن
        self.optimizer.step()

        # إرجاع قيمة الخسارة للمراقبة
        return combined_loss.item()
    # End _update_policy method

    def train(self, num_episodes=1000, print_every=100):
        """
        تنفيذ حلقة التدريب الرئيسية للوكيل.

        Args:
            num_episodes (int): عدد الحلقات المراد تدريب الوكيل خلالها.
            print_every (int): عدد الحلقات بين كل طباعة لتقدم التدريب.

        Returns:
            tuple: يحتوي على (all_total_taus, losses)
                   all_total_taus (list): قائمة بإجمالي Tau لكل حلقة.
                   losses (list): قائمة بمتوسط الخسارة لكل حلقة.
        """
        # قوائم لتخزين سجلات الأداء
        all_total_taus = []
        losses = []
        print(f"Starting training for {num_episodes} episodes...")

        # حلقة التدريب عبر الحلقات
        for episode in range(num_episodes):
            # إعادة تعيين البيئة والحالة الأولية
            state = self.env.reset()
            total_episode_tau = 0.0
            done = False
            step_count = 0
            episode_loss = 0.0
            update_count = 0
            # تحديد حد أقصى للخطوات لمنع الحلقات اللانهائية
            max_steps = self.env.size * self.env.size * 4

            # حلقة الخطوات داخل الحلقة الواحدة
            while not done and step_count < max_steps:
                # 1. اختيار الإجراء والمخاطرة المتوقعة
                action, predicted_risk = self.select_action(state)

                # 2. تنفيذ الإجراء في البيئة
                next_state, reward, done, info = self.env.step(action)

                # 3. حساب Tau وتخزين التجربة
                self.store_experience(state, action, reward, next_state, done, predicted_risk)

                # 4. تحديث سياسة الشبكة (إذا كانت الذاكرة كافية)
                loss_value = self._update_policy()
                if loss_value is not None:
                    # تتبع مجموع الخسارة وعدد التحديثات في الحلقة
                    episode_loss = episode_loss + loss_value
                    update_count = update_count + 1
                # End if statement

                # 5. الانتقال للحالة التالية
                state = next_state

                # 6. تحديث إجمالي Tau المحسوب للحلقة (للمراقبة فقط)
                progress = max(0.0, reward)
                safe_predicted_risk = np.clip(predicted_risk, self.tau_epsilon, 1.0 - self.tau_epsilon)
                denominator = safe_predicted_risk + 0.1 + self.tau_epsilon
                current_tau = (progress + 0.1) / denominator
                total_episode_tau = total_episode_tau + current_tau

                # 7. زيادة عداد الخطوات
                step_count = step_count + 1
            # End while loop (steps in episode)

            # تخزين سجلات الحلقة
            all_total_taus.append(total_episode_tau)
            avg_episode_loss = 0.0
            if update_count > 0:
                 # حساب متوسط الخسارة للحلقة
                 avg_episode_loss = episode_loss / update_count
            # End if statement
            losses.append(avg_episode_loss) # قد يحتوي على أصفار إذا لم تحدث تحديثات


            # طباعة التقدم كل فترة محددة
            is_print_episode = (episode + 1) % print_every == 0
            if is_print_episode:
                # حساب متوسطات آخر 'print_every' حلقة
                last_taus = all_total_taus[-print_every:]
                avg_tau = np.mean(last_taus)
                last_losses = losses[-print_every:]
                # حساب متوسط الخسارة مع تجاهل القيم غير الصالحة (None)
                valid_losses = [l for l in last_losses if l is not None]
                if valid_losses:
                     avg_loss = np.mean(valid_losses)
                else:
                     avg_loss = 0.0 # أو يمكن استخدام np.nan
                # End if/else
                # طباعة ملخص الأداء
                print(f"Episode {episode + 1}/{num_episodes}, Steps: {step_count}, Total Tau: {total_episode_tau:.2f}, Avg Tau (last {print_every}): {avg_tau:.2f}, Avg Loss: {avg_loss:.4f}")
            # End if statement for printing progress
        # End for loop (episodes)

        print("Training finished.")
        # إرجاع سجلات الأداء للتحليل أو الرسم البياني
        return all_total_taus, losses
    # End train method
# End TauAgent class

# ========================================
# 5. التشغيل والاختبار
# ========================================
if __name__ == "__main__":
    # -----------------------------
    # --- إعدادات التجربة ---
    # -----------------------------
    GRID_SIZE = 5           # حجم الشبكة
    HIDDEN_DIM = 64         # حجم الطبقات المخفية
    LEARNING_RATE = 5e-4    # معدل التعلم (مناسب لـ Adam)
    NUM_EPISODES = 2000     # عدد حلقات التدريب
    PRINT_EVERY = 100       # الفاصل الزمني لطباعة التقدم
    BATCH_SIZE = 64         # حجم الدفعة للتحديث
    MEMORY_SIZE = 10000     # حجم ذاكرة إعادة التشغيل
    CLIP_GRAD_NORM = 1.0    # قيمة تقليم التدرج (1.0 قيمة شائعة)
    WEIGHT_DECAY = 1e-5     # معامل تخفيف الوزن L2 (تنظيم بسيط)
    ENTROPY_COEFF = 0.01    # معامل تنظيم الإنتروبيا (لتشجيع الاستكشاف) <--- **تمت زيادته قليلاً**
    USE_CHAOS_OPTIMIZER = False # استخدام Adam حاليًا لأنه أثبت استقراره

    # -----------------------------
    # --- تهيئة البيئة والوكيل ---
    # -----------------------------
    env = AdvancedGridWorld(size=GRID_SIZE)
    agent = TauAgent(env,
                     hidden_dim=HIDDEN_DIM,
                     learning_rate=LEARNING_RATE,
                     batch_size=BATCH_SIZE,
                     memory_size=MEMORY_SIZE,
                     clip_grad_norm=CLIP_GRAD_NORM,
                     use_chaos_optimizer=USE_CHAOS_OPTIMIZER,
                     weight_decay=WEIGHT_DECAY,
                     entropy_coeff=ENTROPY_COEFF) # تمرير معامل الإنتروبيا

    # -----------------------------
    # --- بدء التدريب ---
    # -----------------------------
    tau_history, loss_history = agent.train(num_episodes=NUM_EPISODES, print_every=PRINT_EVERY)

    # -----------------------------
    # --- اختبار الوكيل المدرب ---
    # -----------------------------
    print("\n--- Testing the trained agent ---")
    num_test_episodes = 5 # عدد حلقات الاختبار
    max_test_steps = env.size * env.size * 2 # حد أقصى لخطوات الاختبار

    for i in range(num_test_episodes):
        state = env.reset()
        done = False
        print(f"\nTest Episode {i+1}")
        env.render() # عرض الحالة الأولية
        step = 0
        total_reward = 0
        # حلقة خطوات الاختبار
        while not done and step < max_test_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # اختيار الإجراء الأفضل (الجشع) بدون استكشاف
            with torch.no_grad():
                # الحصول على الاحتمالات والمخاطرة
                _, probs, risk = agent.policy_net(state_tensor)
            # End with block

            # التحقق من صلاحية الاحتمالات
            probs_are_finite = torch.isfinite(probs)
            probs_are_non_neg = probs >= 0
            if not torch.all(probs_are_finite) or not torch.all(probs_are_non_neg):
                 print("Warning: Invalid probs during testing. Choosing random action.")
                 action = random.randrange(agent.action_dim)
                 predicted_risk = 0.5
            else:
                 # اختيار الإجراء ذو الاحتمالية الأعلى
                 action_tensor = torch.argmax(probs)
                 action = action_tensor.item()
                 # الحصول على المخاطرة المتوقعة (مع التحقق من الصلاحية)
                 risk_is_finite = torch.isfinite(risk)
                 if risk_is_finite:
                     predicted_risk = risk.item()
                 else:
                     predicted_risk = 0.5
                 # End if/else
            # End if/else block checking prob validity

            # تنفيذ الخطوة في البيئة
            next_state, reward, done, info = env.step(action)
            total_reward = total_reward + reward
            state = next_state

            # عرض معلومات الخطوة
            action_map = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
            action_name = action_map.get(action, 'Invalid')
            print(f"Step: {step+1}, Action: {action_name}, Reward: {reward:.1f}, Predicted Risk: {predicted_risk:.3f}")
            env.render() # عرض البيئة بعد الخطوة
            step = step + 1
        # End while loop (steps)

        # طباعة نتيجة حلقة الاختبار
        final_pos = env.agent_pos
        goal_reached = done and (final_pos == env.goal)
        if goal_reached:
            print(f"Goal Reached! Total Reward: {total_reward:.2f} in {step} steps.")
        elif done:
            # انتهت الحلقة لسبب آخر (مثل الاصطدام)
            print(f"Episode ended. Final Pos: {final_pos}, Total Reward: {total_reward:.2f} in {step} steps.")
        else:
            # انتهت المهلة الزمنية
            print(f"Episode timed out. Final Pos: {final_pos}, Total Reward: {total_reward:.2f} in {step} steps.")
        # End if/elif/else block reporting episode result
    # End for loop (test episodes)

    print("\nTesting finished.")

    # -----------------------------
    # --- الرسم البياني (اختياري) ---
    # -----------------------------
    try:
        # استيراد المكتبات داخل الكتلة لتجنب الخطأ إذا لم تكن مثبتة
        import matplotlib.pyplot as plt
        import pandas as pd # لاستخدام المتوسط المتحرك بسهولة

        # إنشاء الشكل والمحاور
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # رسم Tau على المحور الأيسر (ax1)
        color_tau = 'tab:red'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Tau', color=color_tau)
        ax1.plot(tau_history, color=color_tau, alpha=0.6, label='Total Tau per Episode')
        # حساب ورسم المتوسط المتحرك لـ Tau
        tau_series = pd.Series(tau_history)
        window_size_tau = min(PRINT_EVERY // 2, len(tau_series)) # حجم نافذة المتوسط المتحرك
        if window_size_tau > 0 :
            moving_avg_tau = tau_series.rolling(window=window_size_tau, min_periods=1).mean()
            ax1.plot(moving_avg_tau, color=color_tau, linestyle='-', linewidth=2, label=f'Tau Moving Avg (w={window_size_tau})')
        # End if statement
        ax1.tick_params(axis='y', labelcolor=color_tau)
        ax1.legend(loc='upper left') # وضع مفتاح الرسم البياني لـ Tau

        # إنشاء محور y ثانٍ للخسارة (ax2) يشارك نفس محور x
        ax2 = ax1.twinx()
        color_loss = 'tab:blue'
        ax2.set_ylabel('Average Loss', color=color_loss)
        # معالجة بيانات الخسارة
        loss_series = pd.Series(loss_history)
        valid_loss_series = loss_series.dropna() # إزالة القيم غير الصالحة (None)
        if not valid_loss_series.empty:
             # حساب ورسم المتوسط المتحرك للخسارة
             window_size_loss = min(PRINT_EVERY // 2, len(valid_loss_series))
             if window_size_loss > 0:
                   moving_avg_loss = valid_loss_series.rolling(window=window_size_loss, min_periods=1).mean()
                   # استخدام الفهرس الأصلي للمحاذاة الصحيحة
                   ax2.plot(moving_avg_loss.index, moving_avg_loss, color=color_loss, linestyle='--', linewidth=2, label=f'Loss Moving Avg (w={window_size_loss})')
             else:
                   # رسم الخسارة الخام إذا كانت النافذة صغيرة جدًا
                   ax2.plot(valid_loss_series.index, valid_loss_series, color=color_loss, linestyle=':', linewidth=1, label='Raw Loss')
             # End if/else statement for loss window size
             # يمكن تفعيل المقياس اللوغاريتمي إذا كانت قيم الخسارة تختلف بشكل كبير
             # ax2.set_yscale('log')
        else:
             # رسالة أو خط فارغ إذا لم تكن هناك بيانات خسارة صالحة
             ax2.plot([], label='Loss Moving Avg (No valid data)')
        # End if/else statement checking for valid loss data
        ax2.tick_params(axis='y', labelcolor=color_loss)
        ax2.legend(loc='upper right') # وضع مفتاح الرسم البياني للخسارة

        # ضبط التخطيط النهائي للرسم البياني
        fig.tight_layout() # منع التداخل بين العناصر
        # إضافة عنوان وصفي للرسم البياني
        title_str = f'Training Progress ({agent.optimizer_type} Optimizer, Entropy Coeff: {agent.entropy_coeff})'
        plt.title(title_str)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6) # إضافة شبكة للمحور y
        plt.show() # عرض الرسم البياني
    except ImportError:
        # رسالة إذا لم تكن مكتبات الرسم البياني مثبتة
        print("\nMatplotlib or Pandas not found. Skipping plot generation.")
    except Exception as e:
        # التعامل مع أي أخطاء أخرى أثناء الرسم
        print(f"\nError during plotting: {e}")
    # End try/except block for plotting

# End main execution block
