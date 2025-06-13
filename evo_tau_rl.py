# -*- coding: utf-8 -*-
"""
==============================================================================
 EvoTauRL: Risk-Aware Reinforcement Learning with Evolving Equations
==============================================================================

**Project:** EvoTauRL (Evolving Tau Reinforcement Learning)
**Version:** 1.0 (Stable with Adam, Evolving Layers Simplified)
**Author (Core Ideas & Initial Code):** [BASIL YAHYA ABDULLAH ]
**AI Assistant (Code Refinement & Implementation):**  AI (via user interaction)




------------------------------------------------------------------------------
                                **مقدمة**
------------------------------------------------------------------------------
هذا المشروع يقدم نظام تعلم معزز مبتكر يهدف إلى حل مشكلة التنقل في بيئة
شبكية (GridWorld) مع الأخذ في الاعتبار مفهوم المخاطرة المتوقعة. يعتمد النظام
على شبكة عصبية ذات بنية فريدة وآلية تعلم مخصصة مستوحاة من الأفكار الأصلية
للمطور [باسل يحيى عبدالله]. تم تطوير الكود وتنقيحه بمساعدة الذكاء الاصطناعي
استنادًا إلى هذه الأفكار الأولية واعتماداً على كود أولي قدمه المطور.

------------------------------------------------------------------------------
                            **الأفكار المبتكرة**
------------------------------------------------------------------------------
يتميز هذا النظام بتطبيق الأفكار المبتكرة التالية، العائدة للمطور [المبتكر العلمي/ باسل يحيى عبدالله]:

1.  **طبقة المعادلة المتطورة (Evolving Equation Layer):**
    بدلاً من استخدام طبقات عصبونية ذات دوال تنشيط ثابتة (مثل ReLU أو Tanh)،
    تُستخدم طبقات يمكنها تكييف بنيتها الرياضية الداخلية. كل طبقة تتعلم:
    *   **معاملات ديناميكية (α, β, γ, δ):** للتحكم في التحويلات الخطية قبل
        وبعد تطبيق الدوال الأساسية.
    *   **اختيار/مزج الدوال:** تتعلم أوزانًا لاختيار أو دمج مجموعة من الدوال
        الأساسية (مثل tanh, relu, identity) بشكل ديناميكي لكل وحدة عصبونية،
        مما يسمح للشبكة باكتشاف التمثيل الرياضي الأمثل للمهمة أثناء التدريب.
    *   **الهدف:** خلق شبكة عصبية أكثر مرونة وقدرة على التكيف مع تعقيدات البيئة.

2.  **مقياس Tau للتعلم الموجه بالمخاطر:**
    يتم تقييم أداء الوكيل ليس فقط بناءً على المكافأة، بل باستخدام مقياس Tau
    الذي يوازن بين "التقدم" (المكافأة الإيجابية نحو الهدف) و "المخاطرة المتوقعة"
    (التي يتم تقديرها بواسطة رأس مخصص في الشبكة العصبية).
    المعادلة المفاهيمية:
        Tau ≈ (التقدم + c1) / (المخاطرة المتوقعة + c2)
    هذا المقياس يوجه عملية التعلم نحو إيجاد مسارات فعالة وآمنة.

3.  **دالة خسارة مجمعة (Risk + Actor + Entropy):**
    يتم تدريب الشبكة باستخدام دالة خسارة متعددة الأهداف:
    *   **خسارة المخاطرة:** تدفع الشبكة لتعلم تقديرات دقيقة للمخاطر وتعديل
        السلوك بناءً على قيم Tau التاريخية (باستخدام صيغة مثل risk * tau).
    *   **خسارة الممثل (Actor):** تُحسِّن سياسة اختيار الإجراءات لتعظيم قيم Tau
        المتوقعة (باستخدام صيغة تشبه Policy Gradient مع Tau كميزة).
    *   **خسارة الإنتروبيا:** تُضاف كمنظم لتشجيع الاستكشاف ومنع التقارب المبكر
        إلى سياسات حتمية قد تكون دون المستوى الأمثل.

------------------------------------------------------------------------------
                            **ملاحظات إضافية**
------------------------------------------------------------------------------
*   **المكونات السابقة (اختيارية):** تضمنت الأفكار الأولية أيضًا "طبقة ارتباط
    ديناميكي" و "انتباه زائدي". تم إزالتها مؤقتًا في هذا الإصدار لتبسيط النموذج
    والتركيز على طبقة المعادلة المتطورة، ولكن يمكن إعادة دمجها في المستقبل.
*   **محسن الشواش:** تم اقتراح "محسن شواش" تجريبي كبديل لـ Adam. نظرًا لمشاكل
    الاستقرار، يُستخدم Adam حاليًا كخيار افتراضي موثوق.

------------------------------------------------------------------------------
                                **الترخيص**
------------------------------------------------------------------------------

------------------------------------------------------------------------------
                            **إخلاء مسؤولية**
------------------------------------------------------------------------------
تم تطوير هذا الكود لأغراض البحث والتجريب. المطور الأصلي [باسل يحيى عبدالله]
ومساعد الذكاء الاصطناعي غير مسؤولين عن أي نتائج أو أضرار مباشرة أو غير مباشرة
قد تنشأ عن استخدام هذا الكود أو الأفكار المطبقة فيه. يقع على عاتق المستخدم
التحقق من ملاءمة الكود لأغراضه الخاصة وتقييم أي مخاطر مرتبطة باستخدامه.

------------------------------------------------------------------------------
                             **الاعتماديات**
------------------------------------------------------------------------------
- Python 3.x
- PyTorch (tested with version X.Y.Z) # اذكر النسخة إن أمكن
- NumPy
- Matplotlib (اختياري، للرسم البياني)
- Pandas (اختياري، للرسم البياني)

==============================================================================
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
import math
import sys
import torch.nn.functional as F
import torch.distributions

# ========================================
# 1. البيئة: GridWorld المتقدمة
# (الكود التفصيلي للبيئة كما هو في الإصدار السابق الصحيح)
# ========================================
class AdvancedGridWorld:
    """
    يمثل بيئة عالم شبكي (GridWorld) ثنائية الأبعاد مع عقبات وهدف.
    (راجع التوثيق السابق لهذه الفئة لمزيد من التفاصيل).
    """
    def __init__(self, size=5):
        self.size = size
        self.goal = (size - 1, size - 1)
        self.obstacles = [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)]
        if self.goal in self.obstacles:
            self.obstacles.remove(self.goal)
        start_pos = (0, 0)
        while start_pos in self.obstacles or start_pos == self.goal:
            start_pos = (random.randint(0, size-1), random.randint(0, size-1))
        self.agent_pos = start_pos
        self.start_pos = start_pos
        self.action_space_n = 4

    def reset(self):
        self.agent_pos = self.start_pos
        return self._get_state()

    def _get_state(self):
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        if 0 <= self.agent_pos[0] < self.size and 0 <= self.agent_pos[1] < self.size:
             grid[self.agent_pos] = 1.0
        if 0 <= self.goal[0] < self.size and 0 <= self.goal[1] < self.size:
             grid[self.goal] = 0.5
        for obs_x, obs_y in self.obstacles:
            if 0 <= obs_x < self.size and 0 <= obs_y < self.size:
                grid[obs_x, obs_y] = -1.0
        return grid.flatten()

    def step(self, action):
        current_x, current_y = self.agent_pos
        next_x, next_y = current_x, current_y
        if action == 0: next_x = max(0, current_x - 1)
        elif action == 1: next_y = min(self.size - 1, current_y + 1)
        elif action == 2: next_x = min(self.size - 1, current_x + 1)
        elif action == 3: next_y = max(0, current_y - 1)
        new_pos = (next_x, next_y)
        reward, done = -0.1, False
        if new_pos == self.goal:
            reward, done = 10.0, True
            self.agent_pos = new_pos
        elif new_pos in self.obstacles:
            reward = -5.0
            # Stay in place
        else:
             self.agent_pos = new_pos
        return self._get_state(), reward, done, {}

    def render(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        if 0 <= self.goal[0] < self.size and 0 <= self.goal[1] < self.size: grid[self.goal[0]][self.goal[1]] = 'G'
        for r, c in self.obstacles:
            if 0 <= r < self.size and 0 <= c < self.size: grid[r][c] = 'X'
        if 0 <= self.agent_pos[0] < self.size and 0 <= self.agent_pos[1] < self.size: grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        print("-" * (self.size * 2 + 1))
        for row in grid: print("|" + " ".join(row) + "|")
        print("-" * (self.size * 2 + 1))
        print(f"Agent @ {self.agent_pos}, Goal @ {self.goal}\n")
# End AdvancedGridWorld class


# ========================================
# 2. بناء المكونات المبتكرة
# ========================================

# ----------------------------------------
# 2.أ. طبقة المعادلة المتطورة [ابتكار جديد]
# ----------------------------------------
class EvolvingEquationLayer(nn.Module):
    """
    [ابتكار 1 - مبسط] طبقة تمثل معادلة رياضية تتطور معاملاتها وتختار دالتها ديناميكيًا.
    تستخدم مجموعة مبسطة من الدوال (tanh, relu, identity).
    الصيغة التقريبية: y ≈ α * SelectedFunc(β @ x + γ) + δ
    """
    def __init__(self, input_dim, output_dim, available_funcs=['tanh', 'relu', 'identity']):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.available_funcs = available_funcs
        self.num_funcs = len(available_funcs)

        # المعاملات الديناميكية القابلة للتعلم (مع تهيئة أولية معقولة)
        self.alpha = nn.Parameter(torch.ones(output_dim))
        self.beta = nn.Parameter(torch.randn(output_dim, input_dim) * np.sqrt(1. / input_dim)) # Kaiming/Xavier like init scaled
        self.gamma = nn.Parameter(torch.zeros(output_dim))
        self.delta = nn.Parameter(torch.zeros(output_dim))

        # أوزان اختيار الدالة (logits)
        self.func_logits = nn.Parameter(torch.zeros(output_dim, self.num_funcs)) # Start with uniform preference

        # قاموس الدوال
        self.func_map = {'tanh': torch.tanh, 'relu': F.relu, 'identity': lambda x: x}
        for func_name in available_funcs:
            if func_name not in self.func_map:
                raise ValueError(f"Function '{func_name}' not found in func_map.")
    # End __init__ method

    def forward(self, x):
        batch_size = x.shape[0]
        # 1. التحويل الخطي + الإزاحة (قبل الدالة)
        transformed_x = F.linear(x, self.beta, self.gamma)

        # 2. تطبيق مزيج مرجح من الدوال
        # استخدام درجة حرارة لزيادة حدة الاختيار (اختياري، يمكن ضبطها)
        temperature = 1.0 # temperature=1.0 means standard softmax
        func_selection_weights = F.softmax(self.func_logits / temperature, dim=-1)

        # تطبيق الدوال
        pre_activation_expanded = transformed_x.unsqueeze(-1)
        all_func_outputs = torch.zeros(batch_size, self.output_dim, self.num_funcs, device=x.device)
        for i, func_name in enumerate(self.available_funcs):
            selected_func = self.func_map[func_name]
            all_func_outputs[:, :, i] = selected_func(pre_activation_expanded).squeeze(-1)
        # End for loop

        # حساب المتوسط المرجح
        expanded_weights = func_selection_weights.unsqueeze(0).expand(batch_size, -1, -1)
        weighted_func_output = torch.sum(expanded_weights * all_func_outputs, dim=-1)

        # 3. التحجيم والإزاحة النهائية
        final_output = self.alpha.unsqueeze(0) * weighted_func_output + self.delta.unsqueeze(0)

        # --- فحص NaN/Inf للمخرجات النهائية للطبقة ---
        if not torch.isfinite(final_output).all():
             print(f"!!! WARNING: NaN/Inf detected in EvolvingEquationLayer output! (Input shape: {x.shape})")
             # محاولة استبدال القيم غير الصالحة بالصفر كإجراء وقائي
             final_output = torch.nan_to_num(final_output, nan=0.0, posinf=1.0, neginf=-1.0)
        # End if statement

        return final_output
    # End forward method

    def get_selected_function_info(self, threshold=0.5):
        """ يعرض معلومات عن الدوال التي تميل الطبقة لاختيارها. """
        info = {}
        with torch.no_grad():
            final_weights = F.softmax(self.func_logits, dim=-1).cpu().numpy()
            for i in range(self.output_dim):
                output_info = {}
                dominant_weight = -1.0
                dominant_func_idx = -1
                for j, func_name in enumerate(self.available_funcs):
                    weight = final_weights[i, j]
                    if weight >= threshold:
                         output_info[func_name] = f"{weight:.2f}"
                    # End if
                    # Track dominant function even if below threshold
                    if weight > dominant_weight:
                        dominant_weight = weight
                        dominant_func_idx = j
                    # End if
                # End inner for loop
                if not output_info and dominant_func_idx != -1: # If nothing above threshold, show the dominant one
                    dominant_func_name = self.available_funcs[dominant_func_idx]
                    output_info[dominant_func_name] = f"{dominant_weight:.2f} (Dominant)"
                # End if
                info[f"Output_{i}"] = output_info if output_info else {"N/A": "Weights too low"}
            # End outer for loop
        # End with block
        return info
    # End get_selected_function_info method
# End EvolvingEquationLayer class


# ========================================
# 3. الشبكة العصبية المبتكرة (مع معادلات متطورة مبسطة)
# ========================================
class EvoTauPolicyNetwork(nn.Module):
    """
    [ابتكار 1 مطبق هنا] شبكة السياسة التي تستخدم طبقات المعادلة المتطورة [المبسطة].
    """
    def __init__(self, input_dim, hidden_dims, action_dim):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # بناء الطبقات المخفية
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            # استخدام الطبقة المتطورة المبسطة
            layers.append(EvolvingEquationLayer(prev_dim, h_dim, available_funcs=['tanh', 'relu', 'identity']))
            # إضافة LayerNorm للاستقرار
            layers.append(nn.LayerNorm(h_dim))
            prev_dim = h_dim
        # End for loop
        self.hidden_layers = nn.Sequential(*layers)

        # رؤوس الشبكة
        self.actor_head = nn.Linear(prev_dim, action_dim)
        self.risk_head = nn.Linear(prev_dim, 1)
        self.epsilon = 1e-8

        # تهيئة انحياز المخاطر
        with torch.no_grad():
            self.risk_head.bias.fill_(-2.0)
    # End __init__ method

    def forward(self, state):
        # تحويل وتجهيز المدخلات
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        is_batched = state.dim() > 1
        if not is_batched:
            state = state.unsqueeze(0)
        state = state.float()

        # المرور عبر الطبقات المخفية
        hidden_output = self.hidden_layers(state)

        # التحقق من NaN/Inf بعد الطبقات المخفية
        if not torch.isfinite(hidden_output).all():
            print(f"!!! WARNING: NaN/Inf detected after hidden layers! Applying nan_to_num.")
            hidden_output = torch.nan_to_num(hidden_output, nan=0.0, posinf=1.0, neginf=-1.0)
        # End if statement

        # حساب مخرجات الرؤوس
        action_logits = self.actor_head(hidden_output)
        risk_logits = self.risk_head(hidden_output)

        # --- عمليات التحقق من الصحة للمخرجات ---
        # التحقق من action_logits
        if not torch.isfinite(action_logits).all():
             print(f"!!! WARNING: NaN/Inf detected in action_logits! Applying nan_to_num.")
             action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        # End if statement

        # حساب الاحتمالات
        action_probs = torch.softmax(action_logits, dim=-1)
        # التحقق من الاحتمالات
        if not torch.isfinite(action_probs).all() or torch.any(action_probs < 0) or \
           not torch.allclose(action_probs.sum(dim=-1), torch.tensor(1.0, device=action_probs.device), atol=1e-5):
            print(f"!!! WARNING: Invalid action_probs calculated! Using uniform distribution as fallback.")
            action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]
            action_logits = torch.zeros_like(action_logits) # Reset logits too
        # End if statement

        # حساب المخاطرة
        risk = torch.sigmoid(risk_logits)
        # التحقق من المخاطرة
        if not torch.isfinite(risk).all():
             print(f"!!! WARNING: NaN/Inf detected in risk output! Clamping.")
             risk = torch.nan_to_num(risk, nan=0.5, posinf=1.0, neginf=0.0)
        # End if statement
        # Clamp risk to valid range
        risk = torch.clamp(risk, min=self.epsilon, max=1.0-self.epsilon)

        # إزالة بُعد الدفعة إذا لزم الأمر
        if not is_batched:
            action_logits = action_logits.squeeze(0)
            action_probs = action_probs.squeeze(0)
            risk = risk.squeeze(0).squeeze(-1) # Squeeze feature dim too
        # End if statement

        return action_logits, action_probs, risk
    # End forward method

    def display_evolving_eq_info(self, threshold=0.5):
        """يعرض معلومات عن الدوال التي تميل كل طبقة متطورة لاختيارها."""
        print("\n--- Evolving Equation Layer Info ---")
        layer_index = 0
        for module in self.hidden_layers:
            if isinstance(module, EvolvingEquationLayer):
                print(f"Evolving Layer {layer_index}:")
                info = module.get_selected_function_info(threshold=threshold)
                for output_name, funcs in info.items():
                    print(f"  {output_name}: {funcs}")
                # End inner for loop
                layer_index += 1
            # End if
        # End outer for loop
        if layer_index == 0:
            print("No EvolvingEquationLayer found in the hidden layers.")
        # End if
        print("------------------------------------\n")
    # End display_evolving_eq_info
# End EvoTauPolicyNetwork class


# ========================================
# 4. خوارزمية التدريب (الوكيل)
# ========================================
class TauAgent:
    """
    الوكيل الذي يستخدم شبكة EvoTauPolicyNetwork ويتعلم باستخدام مقياس Tau
    وخسارة مجمعة (Risk + Actor + Entropy).
    [ابتكار 2 و 3 مطبقان هنا في آلية التعلم].
    """
    def __init__(self, env, hidden_dims, learning_rate=1e-4, gamma=0.99, memory_size=10000, batch_size=64, clip_grad_norm=1.0, use_chaos_optimizer=False, weight_decay=0, entropy_coeff=0.01, actor_loss_weight=1.0):
        self.env = env
        self.input_dim = env.size * env.size
        self.action_dim = env.action_space_n
        # استخدام الشبكة المتطورة
        self.policy_net = EvoTauPolicyNetwork(self.input_dim, hidden_dims, self.action_dim)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau_epsilon = 1e-8
        self.clip_grad_norm = clip_grad_norm
        self.entropy_coeff = entropy_coeff
        self.actor_loss_weight = actor_loss_weight

        # اختيار المُحسِّن (Adam هو الافتراضي الموصى به)
        if use_chaos_optimizer:
            self.optimizer_type = "Chaos"
            effective_lr = min(learning_rate, 1e-5)
            print(f"--- Using {self.optimizer_type} Optimizer with LR: {effective_lr} ---")
            self.optimizer = ChaosOptimizer(self.policy_net.parameters(), lr=effective_lr, weight_decay=weight_decay)
        else:
            self.optimizer_type = "Adam"
            print(f"--- Using {self.optimizer_type} Optimizer with LR: {learning_rate} ---")
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # End if/else block
    # End __init__ method

    def select_action(self, state):
        """ اختيار الإجراء بناءً على سياسة الشبكة (مع استكشاف). """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            _, probs, risk = self.policy_net(state_tensor)

        # التحقق من صلاحية الاحتمالات
        if not torch.isfinite(probs).all() or torch.any(probs < 0):
            probs_np = probs.detach().cpu().numpy()
            print(f"!!! ERROR: Invalid probabilities before sampling: {probs_np}. Choosing random action.")
            action_item = random.randrange(self.action_dim)
            risk_item = risk.item() if torch.isfinite(risk) else 0.5
        else:
            # التأكد من أن المجموع قريب من 1 (قد يحتاج لإعادة تطبيع طفيفة)
            probs_sum = probs.sum()
            if not torch.allclose(probs_sum, torch.tensor(1.0, device=probs.device), atol=1e-5):
                probs = probs / (probs_sum + self.tau_epsilon) # Renormalize

            try:
                # أخذ عينة من التوزيع الاحتمالي
                action_dist = torch.distributions.Categorical(probs=probs)
                action = action_dist.sample()
                action_item = action.item()
                risk_item = risk.item()
            except Exception as e: # Catch broader exceptions during sampling
                print(f"!!! ERROR during sampling: {e}")
                probs_np = probs.detach().cpu().numpy()
                print(f"Probabilities causing error: {probs_np}")
                action_item = random.randrange(self.action_dim) # Fallback
                risk_item = risk.item() if torch.isfinite(risk) else 0.5
            # End try/except
        # End if/else checking prob validity

        return action_item, risk_item
    # End select_action method

    def store_experience(self, state, action, reward, next_state, done, risk):
        """ [ابتكار 2 مطبق هنا] حساب Tau وتخزين التجربة. """
        progress = max(0.0, reward)
        safe_risk = np.clip(risk, self.tau_epsilon, 1.0 - self.tau_epsilon)
        denominator = safe_risk + 0.1 + self.tau_epsilon
        tau = (progress + 0.1) / denominator
        experience = (state, action, tau, next_state, done)
        self.memory.append(experience)
    # End store_experience method

    def _update_policy(self):
        """ [ابتكار 3 مطبق هنا] تحديث الشبكة باستخدام الخسارة المجمعة. """
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states = [exp[0] for exp in batch]
        actions = [exp[1] for exp in batch]
        taus = [exp[2] for exp in batch]

        state_batch = torch.tensor(np.array(states), dtype=torch.float32)
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        tau_batch = torch.tensor(taus, dtype=torch.float32).unsqueeze(1)

        # الحصول على مخرجات الشبكة
        action_logits_batch, _, current_risks = self.policy_net(state_batch)

        # --- Risk Loss (risk * tau) ---
        if not torch.isfinite(current_risks).all() or torch.any(current_risks < 0) or torch.any(current_risks > 1):
             print("!!! WARNING: Invalid risks for loss calc. Clamping.")
             current_risks = torch.clamp(current_risks, min=self.tau_epsilon, max=1.0-self.tau_epsilon)
        safe_current_risks = torch.clamp(current_risks, min=self.tau_epsilon, max=1.0 - self.tau_epsilon)
        risk_loss = (safe_current_risks * tau_batch).mean()

        # --- Actor Loss (Policy Gradient style) ---
        policy_dist = torch.distributions.Categorical(logits=action_logits_batch)
        log_probs_learned = policy_dist.log_prob(action_batch.squeeze(1))
        actor_loss = -(log_probs_learned * tau_batch.squeeze(1).detach()).mean()

        # --- Entropy Loss ---
        entropy = policy_dist.entropy().mean()
        # Handle potential NaN in entropy (if logits are extreme)
        if not torch.isfinite(entropy):
            print(f"!!! WARNING: NaN/Inf detected in entropy ({entropy.item()}). Setting entropy loss to 0.")
            entropy_loss = torch.tensor(0.0, device=risk_loss.device) # Assign zero tensor on correct device
        else:
            entropy_loss = -self.entropy_coeff * entropy
        # End if/else

        # --- Combined Loss ---
        combined_loss = risk_loss + self.actor_loss_weight * actor_loss + entropy_loss

        # التحقق من الخسارة النهائية
        if not torch.isfinite(combined_loss):
            loss_item = combined_loss.detach().item()
            print(f"!!! ERROR: Combined Loss is NaN or Inf ({loss_item}). Skipping backpropagation.")
            # Provide more context for debugging
            risk_loss_item = risk_loss.detach().item()
            actor_loss_item = actor_loss.detach().item()
            entropy_loss_item = entropy_loss.detach().item()
            print(f"Factors - RiskL: {risk_loss_item:.4f}, ActorL: {actor_loss_item:.4f}, EntropyL: {entropy_loss_item:.4f}")
            return None
        # End if statement

        # --- خطوة التحسين ---
        self.optimizer.zero_grad()
        combined_loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.clip_grad_norm)
        self.optimizer.step()

        return combined_loss.item()
    # End _update_policy method

    def train(self, num_episodes=1000, print_every=100, display_eq_info_every=500):
        """ التنفيذ لحلقة التدريب الرئيسية. """
        all_total_taus = []
        losses = []
        print(f"Starting training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            state = self.env.reset()
            total_episode_tau = 0.0
            done = False
            step_count = 0
            episode_loss = 0.0
            update_count = 0
            max_steps = self.env.size * self.env.size * 4

            while not done and step_count < max_steps:
                action, predicted_risk = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.store_experience(state, action, reward, next_state, done, predicted_risk)
                loss_value = self._update_policy()
                if loss_value is not None:
                    episode_loss += loss_value
                    update_count += 1
                state = next_state
                progress = max(0.0, reward)
                safe_predicted_risk = np.clip(predicted_risk, self.tau_epsilon, 1.0 - self.tau_epsilon)
                denominator = safe_predicted_risk + 0.1 + self.tau_epsilon
                current_tau = (progress + 0.1) / denominator
                total_episode_tau += current_tau
                step_count += 1
            # End while loop

            all_total_taus.append(total_episode_tau)
            avg_episode_loss = episode_loss / update_count if update_count > 0 else 0.0
            losses.append(avg_episode_loss)

            # طباعة التقدم
            if (episode + 1) % print_every == 0:
                avg_tau = np.mean(all_total_taus[-print_every:])
                valid_losses = [l for l in losses[-print_every:] if l is not None]
                avg_loss = np.mean(valid_losses) if valid_losses else 0.0
                print(f"Episode {episode + 1}/{num_episodes}, Steps: {step_count}, Total Tau: {total_episode_tau:.2f}, Avg Tau (last {print_every}): {avg_tau:.2f}, Avg Loss: {avg_loss:.4f}")
            # End if statement

            # عرض معلومات تطور المعادلات
            if (episode + 1) % display_eq_info_every == 0:
                # Check if policy_net has the display method before calling
                if hasattr(self.policy_net, 'display_evolving_eq_info') and callable(getattr(self.policy_net, 'display_evolving_eq_info')):
                     self.policy_net.display_evolving_eq_info()
                # End if statement
            # End if statement
        # End for loop

        print("Training finished.")
        return all_total_taus, losses
    # End train method
# End TauAgent class


# ========================================
# 5. التشغيل والاختبار
# ========================================
if __name__ == "__main__":
    # --- Configuration ---
    GRID_SIZE = 5
    HIDDEN_DIMS = [64, 32]    # استخدام طبقتين متطورتين
    LEARNING_RATE = 3e-4     # معدل تعلم معدل
    NUM_EPISODES = 3000      # زيادة عدد الحلقات
    PRINT_EVERY = 100
    BATCH_SIZE = 64
    MEMORY_SIZE = 10000
    CLIP_GRAD_NORM = 1.0
    WEIGHT_DECAY = 1e-5
    ENTROPY_COEFF = 0.1      # معامل إنتروبيا أعلى
    ACTOR_LOSS_WEIGHT = 0.5  # وزن أقل لخسارة الممثل
    USE_CHAOS_OPTIMIZER = False # البقاء مع Adam
    DISPLAY_EQ_INFO_EVERY = 500 # عرض معلومات التطور كل 500 حلقة

    # --- Setup ---
    env = AdvancedGridWorld(size=GRID_SIZE)
    agent = TauAgent(env,
                     hidden_dims=HIDDEN_DIMS,
                     learning_rate=LEARNING_RATE,
                     batch_size=BATCH_SIZE,
                     memory_size=MEMORY_SIZE,
                     clip_grad_norm=CLIP_GRAD_NORM,
                     use_chaos_optimizer=USE_CHAOS_OPTIMIZER,
                     weight_decay=WEIGHT_DECAY,
                     entropy_coeff=ENTROPY_COEFF,
                     actor_loss_weight=ACTOR_LOSS_WEIGHT)

    # --- Train ---
    tau_history, loss_history = agent.train(num_episodes=NUM_EPISODES,
                                             print_every=PRINT_EVERY,
                                             display_eq_info_every=DISPLAY_EQ_INFO_EVERY)

    # --- Display final Evolving Equation Info ---
    # Check if policy_net has the display method before calling
    if hasattr(agent.policy_net, 'display_evolving_eq_info') and callable(getattr(agent.policy_net, 'display_evolving_eq_info')):
        agent.policy_net.display_evolving_eq_info()
    # End if statement

    # --- Test ---
    print("\n--- Testing the trained agent ---")
    num_test_episodes = 5
    max_test_steps = env.size * env.size * 2

    for i in range(num_test_episodes):
        state = env.reset()
        done = False
        print(f"\nTest Episode {i+1}")
        env.render()
        step = 0
        total_reward = 0
        while not done and step < max_test_steps:
            # استخدام select_action للاختبار (يعكس السياسة مع الاستكشاف الطفيف)
            action, predicted_risk = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state

            action_map = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
            action_name = action_map.get(action, 'Invalid')
            print(f"Step: {step+1}, Action: {action_name}, Reward: {reward:.1f}, Pred. Risk: {predicted_risk:.3f}")
            env.render()
            step += 1
        # End while loop

        final_pos = env.agent_pos
        goal_reached = done and (final_pos == env.goal)
        if goal_reached:
            print(f"Goal Reached! Total Reward: {total_reward:.2f} in {step} steps.")
        elif done:
            print(f"Episode ended. Final Pos: {final_pos}, Total Reward: {total_reward:.2f} in {step} steps.")
        else:
            print(f"Episode timed out. Final Pos: {final_pos}, Total Reward: {total_reward:.2f} in {step} steps.")
        # End if/elif/else
    # End for loop

    print("\nTesting finished.")

    # --- Plotting (Optional, check title) ---
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:red'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Tau', color=color)
        ax1.plot(tau_history, color=color, alpha=0.6, label='Total Tau per Episode')
        tau_series = pd.Series(tau_history)
        window_size_tau = min(PRINT_EVERY // 2, len(tau_series))
        if window_size_tau > 0 :
            moving_avg_tau = tau_series.rolling(window=window_size_tau, min_periods=1).mean()
            ax1.plot(moving_avg_tau, color=color, linestyle='-', linewidth=2, label=f'Tau Moving Avg (w={window_size_tau})')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Average Loss', color=color)
        loss_series = pd.Series(loss_history)
        valid_loss_series = loss_series.dropna()
        if not valid_loss_series.empty:
             window_size_loss = min(PRINT_EVERY // 2, len(valid_loss_series))
             if window_size_loss > 0:
                   moving_avg_loss = valid_loss_series.rolling(window=window_size_loss, min_periods=1).mean()
                   ax2.plot(moving_avg_loss.index, moving_avg_loss, color=color, linestyle='--', linewidth=2, label=f'Loss Moving Avg (w={window_size_loss})')
             else:
                   ax2.plot(valid_loss_series.index, valid_loss_series, color=color, linestyle=':', linewidth=1, label='Raw Loss')
             # Optional log scale
             # ax2.set_yscale('log')
        else:
             ax2.plot([], label='Loss Moving Avg (No valid data)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        fig.tight_layout()
        # Update title with specific parameters
        title_str = f'EvoTauRL ({agent.optimizer_type} LR:{LEARNING_RATE}, Entr:{ENTROPY_COEFF}, ActW:{ACTOR_LOSS_WEIGHT})'
        plt.title(title_str)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.show()
    except ImportError:
        print("\nMatplotlib or Pandas not found. Skipping plot generation.")
    except Exception as e:
        print(f"\nError during plotting: {e}")
    # End try/except block

# End main execution block
