# 统一 instruction 与源区域协议

### 一条指令与一个源区域

Benchmark 与训练数据共用以下规范：英文语义记录/渲染格式为 `instruction_protocol_version=1.0`，mask 含义契约为 `mask_semantics.version=1.1`，写作风格为 `benchmark-concise-2-direct-add-1`。三者分别描述可序列化字段、区域的意义和语言写法，不是三套不同任务协议。代码与本段正文在两个仓库逐字核对。

每个编辑目标 U 绑定一个源实例或部件 S、一种操作及其指令。Point、box、mask 和文字 ref 是 S 的不同定位形式，互相转换不产生新目标。原始 mask 保留可见像素、遮挡和孔洞；不把 bbox 内未选中的醒目物体误当成 S。Mask 不因编辑改成结果轮廓，也不是允许任意修改其内部全部内容的授权。

实际编辑对象 T 可以等于 S，也可以是 S 内明确命名的部件。若 mask 已指向袖口，直接说“选中的袖口”；若 mask 指向上衣而只编辑袖口，指令必须写明袖口，构造记录必须表达这一层级，不能把 scope 偷换成袖口以绕过执行区域约束。

### 四类指令与 mask 的含义

| 类型 | 标准句式 | 源 mask 的含义 | 必须保持的边界 |
|---|---|---|---|
| add | `Add <item> <relation> <anchor>.` | 现有直接承载物、容器、穿戴对象、安装面或附着部位；少数符合条件的紧邻参照物 | 宿主保留；新增数量、接触、尺度及位置自然；不是新增物最终轮廓 |
| replace | `Replace <target> with <replacement>.` | 被替换的现有物体/部件；宿主选择时显式指定内部部件 | 新旧对应同一对象层级，有真实结构/品类变化；单纯换色归 attribute |
| remove | `Remove <target>.` 或 `Remove <part> from <host>.` | 要删除的现有目标，或包含所述删除部件的宿主 | 只删除指明范围；不把整个宿主清空；不设计不自然的断肢任务 |
| attribute | `Change the <property> of <target> to <value>.` | 属性变化所属的现有物体、表面或明确部件 | 保留身份、结构及未指定内容；不得把“给人改色”含糊代指衣服 |

Attribute 支持 color/material/finish/pattern/texture/text；透明物染色用 `Tint <target> <color>.`，透明度用 `Make <target> transparent/translucent/opaque.`。文字内容逐字保留并加引号；仅必要时在一句话内列多个属性。加实体徽章/布贴属于 add，改变原表面图案属于 attribute，删除已有标志属于 remove。

删除内容物、连同内容物删除、剪短衣物等须写明范围和终点。一个 U 不混入不同操作；同一锚点添加多个同类物品可为一个 U，但必须说明数量。多目标 case 的每个 U 独立绑定指令，其他 U 授权的变化不算该 U 的越界。

### Add：直接锚点优先，安装部位与关系分开

1. 优先在所选对象上直接承载、容纳、穿戴、附着或印制。先根据宿主功能、材质、尺寸和可见空间选择自然的新物，而不是先想好物品再寻找远处参照。
2. `placement.relation` 写新增物与宿主/安装部位的关系，如 `on/in/attached to/mounted on/sewn onto/clipped to/hanging from`；`placement.site` 只写必要的宿主内部位置。`ref` 负责区分哪个宿主，不重复承担 site 的内容。
3. “在袖口下方的衣袖上缝补丁”是直接附着：`relation=sewn onto`、`site=area below the cuff`、`anchor_mode=support`。Site 里出现 below/near 不会把它变成参照添加。“椅子旁的包”用于识别包，给包提手加牌子仍是直接锚点。
4. 新物边缘、必要接触阴影和遮挡可超出源 mask，不能要求全部新增像素都落在原 mask 内。但必须与该宿主保持真实接触/支撑，且不妨碍原有功能、不改动无关实例。
5. 只有直接添加不自然时，才允许 `anchor_mode=reference`：位置真正紧邻、可见支撑面明确、侧别/安装位置唯一。远距离地标、悬浮新物、可落在多个位置的“附近”均不接受。训练构造必须有可信执行区域；缺少时换类型或拒绝该指令，不判原 mask 无效。
6. 本版 benchmark 的全部 add 均使用直接锚点。协议允许紧邻例外，不等于当前样本含此例外。Benchmark 用户只提供源定位，不需要预先画出新物轮廓；训练内部执行 mask 不能冒充 benchmark 输入。

### 精简、准确、无歧义

默认一句祈使句：动作＋最短无歧义目标短语＋必要结果。目标唯一时停止添加限定；有竞争实例时才增加一个经图像核实的所属、方位、颜色或序数。序数必须说明可见计数组和方向；身体左右易混淆时用 image-left/image-right。颜色从无标注原图读取，不从彩色 mask 叠加读取。

不要另起 `which is/located/the one...` 分句解释位置，不堆叠宿主、颜色、方向，不列排除项。不要附加“其他都不变”“保留材质/光照”等通用保留列表：这些由公共契约和内部生成提示承担。`qualifier` 默认空，仅当省略会改变请求语义时保留短限定。必要的计数、精确文字、安装位置和删除范围不可为缩句而删除。

常规目标是 Ref 不超过26个英文空格词、No-ref 不超过16词；不是硬截断阈值，不填充到目标长度。Replace 的新形态写在一个完整替换物短语中，内部 form_change 证据不再追加到句尾。模型在同一次请求内检查并删冗，不默认增加润色调用。

### Ref / No-ref 和结构化记录

两种指令从同一 `instruction_spec` 渲染；No-ref 只把识别源实例的 ref 换成 `the selected <scope>`。操作、部件层级、物品、数量、目标属性、局部安装位置和必要限定必须相同。No-ref 必须有 point/box/mask；ref-only 不泄露私有区域或完整语义字段。

例如 Ref 为 `Add a red tag clipped to the handle of the left suitcase.`，No-ref 为 `Add a red tag clipped to the handle of the selected suitcase.`。若源区域已经选中提手，则 No-ref 是 `Add a red tag clipped to the selected handle.`，不能重复“提手的提手”。

作者记录包含 operation、核实后的 ref/scope、payload、target_part/remove_mode、attributes、placement 与必要 qualifier。Add 的物品 payload 不保留指代宿主的悬空 its/it，连接关系和 site 单独存储；构造器不得靠扫描 near/below/above 猜测锚点语义。英文、中文、completion requirement、edit contract 和摘要必须随同一条作者记录一起更新。

### 训练执行与质量检查

源定位 mask、实际执行区域 execution/support mask、邻近保护区域是不同概念。宿主内子部件及紧邻参照任务必须有可信执行区域；检查区域与语义层级相符，不能仅把 ref 改成部件名称而仍开放整个宿主。区域外像素不变只能证明合成边界，不能证明区域内改对、自然或没有误伤。

检查依次覆盖：原图/mask 绑定及几何；逐条 Ref/No-ref 等价与规范；图像中的实例归属和最接近竞争目标；操作自然性与类别边界；add 接触关系和局部位置；生成后完成度、非目标保护和自然质量。自动校验、模型视觉审核、assistant 目视复核和独立人工验收分别记录，不能互相冒充。
