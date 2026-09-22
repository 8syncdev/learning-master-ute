import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import './style.css'

const P = [
  { id:'A', x:2, y:3 }, { id:'B', x:5, y:18 }, { id:'C', x:8, y:7 },
  { id:'D', x:9, y:8 }, { id:'E', x:13, y:17 }, { id:'F', x:20, y:3 }
]
const splitX = 8.5
const d0 = Math.sqrt(52)
const pseudo = [
  'closestPair(Px, Py):',
  '  if n <= 3: bruteForce(Px)',
  '  chia Px thành Lx và Rx',
  '  tách Py thành Ly và Ry trong O(n)',
  '  left  = closestPair(Lx, Ly)',
  '  right = closestPair(Rx, Ry)',
  '  d = min(left.d, right.d)',
  '  strip = các điểm |x - midX| < d, giữ thứ tự y',
  '  với mỗi điểm p trong strip:',
  '    so p với tối đa 7 điểm kế tiếp',
  '  return cặp tốt nhất',
  'T(n) = 2T(n/2) + O(n) => O(n log n)'
]

const steps = [
  {
    section:'Nền tảng', title:'Điểm và khoảng cách là gì?', subtitle:'Bắt đầu từ số 0', stage:'distance',
    bullets:['Một điểm P(x,y) chỉ là một vị trí trên mặt phẳng.', 'Muốn biết hai điểm gần nhau đến đâu, ta dùng khoảng cách Euclid.', 'Ví dụ A(2,3) và C(8,7): lệch 6 theo x, lệch 4 theo y.'],
    formula:'dist(A,C) = √(6² + 4²) = √52 ≈ 7.21',
    explain:'Hãy tưởng tượng đi từ A sang C: đi ngang 6 ô rồi đi dọc 4 ô. Đường thẳng nối A–C là cạnh huyền, nên công thức đến trực tiếp từ định lý Pythagore.',
    why:'Nếu chưa hiểu công thức khoảng cách thì những bước sau chỉ giống học thuộc. Đây là phép đo mà thuật toán sẽ cố làm ít lần nhất có thể.',
    talk:'Trước khi nói về thuật toán, em chốt một ý rất cơ bản: mỗi cặp điểm có một khoảng cách. Bài toán chỉ hỏi cặp nào có khoảng cách nhỏ nhất.',
    line:0, check:['Nếu hai điểm trùng nhau thì khoảng cách bằng bao nhiêu?','Bằng 0, và đó chắc chắn là khoảng cách nhỏ nhất có thể.']
  },
  {
    section:'Nền tảng', title:'Bài toán cần giải', subtitle:'Tìm cặp gần nhất trong n điểm', stage:'brute',
    bullets:['Với 6 điểm demo có 15 cặp khác nhau.', 'Cách ngây thơ: tính khoảng cách của tất cả 15 cặp rồi lấy nhỏ nhất.', 'Tổng quát n điểm có n(n−1)/2 cặp, nên thời gian tăng theo O(n²).'],
    formula:'số cặp = n(n−1)/2',
    explain:'6 điểm thì 15 cặp chưa đáng ngại. Nhưng 1.000.000 điểm tạo xấp xỉ 500 tỷ cặp. Chính số cặp bùng nổ này là lý do ta cần thuật toán tốt hơn.',
    why:'Mục tiêu của chia để trị không phải làm công thức khoảng cách nhanh hơn. Mục tiêu là tránh phải đo những cặp mà ta biết chắc không thể thắng.',
    talk:'Nếu em vét cạn thì không sai, nhưng dữ liệu lớn sẽ quá chậm. Chia để trị giúp loại bỏ phần lớn cặp không cần kiểm tra.',
    line:1, check:['Brute force sai ở đâu?','Không sai về kết quả. Nó chỉ không hiệu quả khi n lớn vì phải xét O(n²) cặp.']
  },
  {
    section:'Ý tưởng', title:'Chia để trị là gì?', subtitle:'Divide → Conquer → Combine', stage:'concept',
    bullets:['Divide: chia bài toán lớn thành các bài toán cùng loại nhưng nhỏ hơn.', 'Conquer: giải từng bài toán con.', 'Combine: ghép kết quả con và xử lý trường hợp nằm qua ranh giới.'],
    formula:'Bài toán lớn → 2 bài toán nhỏ → ghép kết quả',
    explain:'Giống tìm hai người đứng gần nhau nhất trong một sân đông: trước hết chia sân làm hai khu, tìm cặp gần nhất mỗi khu, rồi chỉ kiểm tra những người đứng sát đường chia.',
    why:'Điểm tinh tế nằm ở Combine. Nếu ta lại so mọi người ở nửa trái với mọi người ở nửa phải thì thuật toán quay về gần O(n²).',
    talk:'Ba từ khóa em cần nhớ là Divide, Conquer, Combine. Với Closest Pair, bước Combine mới là chỗ tạo ra khác biệt.',
    line:0, check:['Vì sao không thể chỉ lấy min của hai nửa rồi dừng?','Vì cặp gần nhất toàn cục có thể gồm một điểm bên trái và một điểm bên phải.']
  },
  {
    section:'Chuẩn bị', title:'Vì sao phải sort trước?', subtitle:'Px theo x, Py theo y', stage:'sorted',
    bullets:['Px giúp chia trái/phải nhanh và ổn định ở mọi tầng đệ quy.', 'Py giúp bước Combine quét theo chiều y mà không phải sort lại.', 'Ta sort một lần ở đầu: O(n log n).'],
    formula:'Px = sortByX(P),  Py = sortByY(P)',
    explain:'Nếu mỗi lần đệ quy lại sort strip theo y, ta tốn thêm log n ở nhiều tầng. Muốn đạt đúng O(n log n), cần mang thứ tự y đi xuống các bài toán con.',
    why:'Đây là chi tiết rất hay bị bỏ qua. Ý tưởng chia để trị đúng nhưng cài đặt sort lại liên tục có thể thành O(n log² n).',
    talk:'Em tạo hai cách nhìn của cùng một tập điểm: một danh sách theo x để chia, một danh sách theo y để gộp nhanh.',
    line:1, check:['Px và Py có phải hai tập điểm khác nhau không?','Không. Chúng chứa đúng cùng các điểm, chỉ khác thứ tự sắp xếp.']
  },
  {
    section:'Divide', title:'Chia tập điểm làm đôi', subtitle:'Đường giữa x = 8.5', stage:'divide',
    bullets:['Theo Px: A, B, C | D, E, F.', 'Nửa trái có 3 điểm, nửa phải có 3 điểm.', 'Py cũng được tách tương ứng sang Ly và Ry trong O(n), không sort lại.'],
    formula:'n → n/2 + n/2',
    explain:'Ta không chia ngẫu nhiên. Chia theo x đảm bảo hai bài toán con cân bằng về số lượng điểm, giúp chiều sâu đệ quy chỉ khoảng log₂n.',
    why:'Nếu một bên có gần n điểm còn bên kia rất ít, ta mất lợi ích của việc chia cân bằng.',
    talk:'Ở bước Divide, em chỉ tách dữ liệu. Chưa kết luận cặp nào gần nhất.',
    line:2, check:['Tại sao chia gần n/2 là tốt?','Vì cây đệ quy nông, chỉ khoảng log₂n tầng thay vì có thể kéo dài gần n tầng.']
  },
  {
    section:'Conquer', title:'Giải nửa trái', subtitle:'A, B, C', stage:'left',
    bullets:['Bài toán con chỉ còn 3 điểm nên dùng brute force.', 'Ba cặp là A–B, A–C, B–C.', 'Nhỏ nhất trong nửa trái là A–C.'],
    formula:'dL = dist(A,C) = √52 ≈ 7.21',
    explain:'Base case n ≤ 3 dùng vét cạn là hợp lý vì số cặp rất ít. Chia tiếp nữa không đem lại lợi ích mà còn làm code phức tạp.',
    why:'Chia để trị không loại bỏ brute force hoàn toàn. Nó dùng brute force đúng nơi brute force rẻ nhất: bài toán con rất nhỏ.',
    talk:'Nửa trái trả lên đúng một thông tin quan trọng: cặp tốt nhất bên trái và khoảng cách dL.',
    line:4, check:['Ta có cần giữ mọi khoảng cách của nửa trái không?','Không. Để ghép, chỉ cần cặp tốt nhất và khoảng cách nhỏ nhất dL.']
  },
  {
    section:'Conquer', title:'Giải nửa phải', subtitle:'D, E, F', stage:'right',
    bullets:['Nửa phải cũng có 3 điểm nên kiểm tra trực tiếp.', 'Cặp gần nhất trong D, E, F là D–E.', 'Kết quả nửa phải được trả lên độc lập với nửa trái.'],
    formula:'dR = dist(D,E) = √97 ≈ 9.85',
    explain:'Hai lời gọi đệ quy giải cùng một loại bài toán. Điểm khác nhau chỉ là dữ liệu đầu vào nhỏ hơn.',
    why:'Đây là đặc trưng của divide and conquer: cấu trúc bài toán con giống hệt bài toán gốc.',
    talk:'Sau Conquer, em có hai ứng viên: tốt nhất bên trái và tốt nhất bên phải.',
    line:5, check:['Lúc này D–E có phải đáp án toàn cục không?','Chưa. Nó chỉ là đáp án tốt nhất của nửa phải.']
  },
  {
    section:'Combine', title:'Tạo ngưỡng d hiện tại', subtitle:'Chọn kết quả tốt hơn của hai nửa', stage:'min',
    bullets:['dL ≈ 7.21, dR ≈ 9.85.', 'Lấy d = min(dL,dR) = 7.21.', 'd là khoảng cách tốt nhất đã biết, chưa chắc là đáp án cuối.'],
    formula:'d = min(7.21, 9.85) = 7.21',
    explain:'Từ đây d trở thành một “hàng rào”: một cặp mới chỉ đáng quan tâm nếu nó có cơ hội nhỏ hơn 7.21.',
    why:'Ngưỡng d giúp ta loại điểm theo hình học mà không cần tính khoảng cách của mọi cặp.',
    talk:'Em coi d là kỷ lục hiện tại. Bất kỳ cặp qua biên nào muốn thắng phải có khoảng cách nhỏ hơn d.',
    line:6, check:['Nếu một cặp có chênh lệch x đã lớn hơn d thì có thể thắng không?','Không. Chỉ riêng độ lệch x đã vượt d, nên khoảng cách Euclid chắc chắn lớn hơn d.']
  },
  {
    section:'Combine', title:'Vì sao chỉ cần xét strip?', subtitle:'Loại những điểm chắc chắn không thể thắng', stage:'strip',
    bullets:['Strip chứa điểm có |x − midX| < d.', 'Điểm nằm xa đường chia ít nhất d không thể tạo cặp qua biên ngắn hơn d.', 'Vì vậy ta chỉ giữ một dải hẹp quanh đường giữa.'],
    formula:'strip = { P | |P.x − 8.5| < 7.21 }',
    explain:'Giả sử điểm bên trái cách đường chia 9 đơn vị trong khi d = 7.21. Dù điểm bên phải nằm sát đường chia đến đâu, riêng khoảng cách theo x đã là 9, nên cặp đó thua chắc.',
    why:'Đây là phép cắt giảm ứng viên quan trọng nhất: từ “mọi điểm trái × mọi điểm phải” xuống chỉ các điểm sát biên.',
    talk:'Strip không phải mẹo tùy ý. Nó xuất phát trực tiếp từ điều kiện muốn có khoảng cách nhỏ hơn d.',
    line:7, check:['Một điểm cách midX đúng bằng 10 khi d=7.21 có cần vào strip không?','Không, vì chỉ riêng chênh lệch x đã lớn hơn khoảng cách tốt nhất hiện tại.']
  },
  {
    section:'Combine', title:'Vì sao strip phải theo y?', subtitle:'Loại tiếp bằng chênh lệch dọc', stage:'stripY',
    bullets:['Strip được lấy từ Py nên đã có thứ tự y.', 'Khi quét từ dưới lên, nếu chênh lệch y ≥ d thì có thể dừng xét xa hơn.', 'Ta không cần so một điểm với toàn bộ strip.'],
    formula:'nếu |y₂ − y₁| ≥ d  ⇒  dist(P₁,P₂) ≥ d',
    explain:'Cũng giống trục x: nếu chỉ riêng độ lệch y đã không nhỏ hơn d thì khoảng cách thật càng không thể nhỏ hơn d.',
    why:'Thứ tự y biến việc tìm ứng viên gần thành một lần quét tuyến tính thay vì tạo thêm một vòng lặp lớn.',
    talk:'Sort theo y cho em một điều kiện dừng rất mạnh: đi lên quá d thì bỏ luôn phần còn lại cho điểm hiện tại.',
    line:7, check:['Vì sao Py quan trọng hơn việc sort strip lại ở mỗi lần gọi?','Vì Py giữ sẵn thứ tự y xuyên suốt đệ quy, giúp combine O(n).']
  },
  {
    section:'Combine', title:'Tại sao chỉ cần tối đa 7 điểm kế tiếp?', subtitle:'Mấu chốt hình học của thuật toán', stage:'neighbors',
    bullets:['Với mỗi điểm trong strip, chỉ xét các điểm tiếp theo có y cách nó < d.', 'Một định lý đóng gói hình học chứng minh trong vùng cần xét không thể có quá 7 ứng viên kế tiếp cần kiểm tra.', 'Quan trọng về độ phức tạp: 7 là hằng số, không tăng theo n.'],
    formula:'mỗi điểm: ≤ 7 phép so sánh ứng viên',
    explain:'Bạn không cần học thuộc chứng minh “7” để thuyết trình cơ bản. Cần hiểu trực giác: vì mỗi nửa đã biết không có hai điểm nào gần hơn d, các điểm trong dải hẹp không thể chen dày tùy ý.',
    why:'Nhờ giới hạn hằng số này, quét strip là O(n), không phải O(n²). Đây là lý do Combine vẫn rẻ.',
    talk:'Điểm em muốn nhấn mạnh không phải con số 7, mà là mỗi điểm chỉ so với một số lượng hằng số hàng xóm gần nó.',
    line:9, check:['Nếu mỗi điểm phải so với n điểm khác trong strip thì Combine là bao nhiêu?','Có thể lên O(n²), và toàn bộ lợi thế chia để trị sẽ mất.']
  },
  {
    section:'Combine', title:'Phát hiện cặp nằm qua biên', subtitle:'C và D thắng kỷ lục hiện tại', stage:'scan',
    bullets:['C(8,7) ở trái, D(9,8) ở phải.', 'Chênh lệch x = 1, chênh lệch y = 1.', 'Khoảng cách √2 ≈ 1.414 < d = 7.21, nên cập nhật đáp án.'],
    formula:'dist(C,D) = √(1² + 1²) = √2 ≈ 1.414',
    explain:'Đây chính là trường hợp khiến ta bắt buộc phải có Combine. Nếu chỉ lấy min(dL,dR), thuật toán sẽ trả A–C = 7.21 và bị sai.',
    why:'Ví dụ được chọn cố ý để cặp đúng nằm ở hai phía đường chia, giúp nhìn thấy vai trò của strip.',
    talk:'Cặp C–D không xuất hiện trong kết quả bên trái hay bên phải. Chỉ bước Combine mới tìm ra nó.',
    line:9, check:['Nếu bỏ Combine, kết quả demo này sẽ là gì?','Sai: thuật toán sẽ trả A–C với 7.21 thay vì C–D với 1.414.']
  },
  {
    section:'Kết quả', title:'Kết luận thuật toán', subtitle:'Closest Pair = C–D', stage:'result',
    bullets:['Cặp gần nhất: C(8,7) và D(9,8).', 'Khoảng cách nhỏ nhất: √2 ≈ 1.414.', 'Kết quả cuối cùng là min của trái, phải và strip.'],
    formula:'answer = min(left, right, cross-strip)',
    explain:'Luồng tư duy hoàn chỉnh là: chia để giảm phạm vi, giải từng nửa để có ngưỡng d, rồi dùng d để kiểm tra biên thật thông minh.',
    why:'Thuật toán đúng vì mọi cặp có thể thuộc một trong ba loại: cùng nửa trái, cùng nửa phải, hoặc nằm qua hai nửa. Ta đã xét đủ cả ba.',
    talk:'Đây là cách em chứng minh trực giác tính đúng: không có loại cặp nào bị bỏ sót.',
    line:10, check:['Ba loại cặp cần phủ đủ là gì?','Trái–trái, phải–phải, và trái–phải qua strip.']
  },
  {
    section:'Phân tích', title:'T(n) hình thành như thế nào?', subtitle:'Hiểu O(n log n), không học thuộc', stage:'complexity',
    bullets:['Mỗi tầng có 2 bài toán con kích thước n/2.', 'Tổng công việc Combine của một tầng là O(n).', 'Có khoảng log₂n tầng vì cứ chia đôi đến bài toán nhỏ.'],
    formula:'T(n) = 2T(n/2) + O(n)  ⇒  O(n log n)',
    explain:'Hình dung cây đệ quy: mỗi tầng dù có nhiều node hơn, tổng số điểm được xử lý vẫn khoảng n. Có log n tầng, nên n × log n.',
    why:'Đây là cách giải thích trực quan hơn Master Theorem. Master Theorem chỉ xác nhận kết quả mà cây đệ quy đã cho ta thấy.',
    talk:'Mỗi tầng tốn tuyến tính, số tầng là log n, nên tổng là n log n. Đây là phần em có thể nói trước rồi mới nhắc Master Theorem.',
    line:11, check:['Vì sao số tầng là log₂n?','Vì sau k lần chia còn n/2^k điểm; dừng khi còn hằng số điểm, nên k xấp xỉ log₂n.']
  },
  {
    section:'Kết thúc', title:'Đánh giá và dùng khi nào?', subtitle:'Biết ưu, nhược và ứng dụng', stage:'use',
    bullets:['Ưu: chính xác, O(n log n), rất phù hợp dữ liệu điểm lớn.', 'Nhược: cài đặt khó hơn O(n²), phải quản lý Px/Py và Combine cẩn thận.', 'Ứng dụng: GIS, robot, game, clustering, hình học tính toán, phát hiện đối tượng gần nhau.'],
    formula:'Time O(n log n) · Extra space O(n)',
    explain:'Với vài chục điểm, brute force thường đủ đơn giản. Khi n lớn hoặc cần nền tảng cho hình học tính toán, divide and conquer đáng dùng.',
    why:'Khuyến nghị thuật toán tốt không chỉ nhìn Big O. Còn phải cân nhắc độ phức tạp cài đặt, kích thước dữ liệu và yêu cầu thực tế.',
    talk:'Nếu thầy hỏi khi nào không nên dùng, em trả lời: dữ liệu rất nhỏ thì brute force dễ cài và đủ nhanh.',
    line:11, check:['Điểm dễ sai nhất khi cài bản O(n log n) là gì?','Sort lại theo y ở mỗi tầng hoặc xử lý strip sai, làm chậm hơn hoặc bỏ sót cặp qua biên.']
  }
]

const W=760, H=430, PAD=54, XMAX=22, YMAX=21
const sx=x=>PAD+(x/XMAX)*(W-PAD*2)
const sy=y=>H-PAD-(y/YMAX)*(H-PAD*2)
const point=id=>P.find(p=>p.id===id)

function Pair({ ids, primary=false, label='' }) {
  const a=point(ids[0]), b=point(ids[1]), mx=(sx(a.x)+sx(b.x))/2, my=(sy(a.y)+sy(b.y))/2
  return <g className={primary?'pair primary':'pair'}>
    <line x1={sx(a.x)} y1={sy(a.y)} x2={sx(b.x)} y2={sy(b.y)} />
    <rect x={mx-30} y={my-17} width="60" height="24" rx="6" />
    <text x={mx} y={my} textAnchor="middle">{label}</text>
  </g>
}

function DacFlow(){
  return <div className="concept-flow">
    <div><b>1</b><strong>DIVIDE</strong><p>Chia tập điểm theo trục x thành hai nửa cân bằng.</p></div>
    <span>→</span>
    <div><b>2</b><strong>CONQUER</strong><p>Tìm cặp gần nhất độc lập trong từng nửa.</p></div>
    <span>→</span>
    <div><b>3</b><strong>COMBINE</strong><p>Kiểm tra dải sát đường chia để không bỏ sót cặp qua biên.</p></div>
  </div>
}

function ComplexityTree(){
  return <div className="tree">
    <div className="tree-note">Mỗi tầng cộng lại vẫn xử lý khoảng <b>n</b> điểm</div>
    <div className="tree-row"><span>n</span><em>O(n)</em></div>
    <i/>
    <div className="tree-row two"><span>n/2</span><span>n/2</span><em>O(n)</em></div>
    <i/>
    <div className="tree-row four"><span>n/4</span><span>n/4</span><span>n/4</span><span>n/4</span><em>O(n)</em></div>
    <div className="tree-dots">⋮ khoảng log₂n tầng ⋮</div>
    <div className="tree-total">O(n) × O(log n) = <strong>O(n log n)</strong></div>
  </div>
}

function Plot({ stage }) {
  const divide=['divide','left','right','min','strip','stripY','neighbors','scan','result'].includes(stage)
  const showL=['left','min','strip','stripY','neighbors','scan'].includes(stage)
  const showR=['right','min','strip','stripY','neighbors','scan'].includes(stage)
  const strip=['strip','stripY','neighbors','scan','result'].includes(stage)
  const final=['scan','result'].includes(stage)
  const left=Math.max(0,splitX-d0), right=Math.min(XMAX,splitX+d0)
  const allPairs=[]
  if(stage==='brute'){
    for(let a=0;a<P.length;a++) for(let b=a+1;b<P.length;b++) allPairs.push([P[a],P[b]])
  }
  const yOrder=[...P].sort((a,b)=>a.y-b.y || a.x-b.x)

  return <svg className="plot" viewBox={'0 0 '+W+' '+H} role="img" aria-label="Mặt phẳng tọa độ minh họa Closest Pair">
    <defs><pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" className="gridline"/></pattern></defs>
    <rect className="plotbg" width={W} height={H} rx="13"/>
    <rect x={PAD} y={PAD} width={W-PAD*2} height={H-PAD*2} fill="url(#grid)"/>
    {allPairs.map(([a,b])=><line key={a.id+b.id} className="brute-line" x1={sx(a.x)} y1={sy(a.y)} x2={sx(b.x)} y2={sy(b.y)}/>)}
    {strip && <rect className="strip" x={sx(left)} y={PAD} width={sx(right)-sx(left)} height={H-PAD*2}/>}
    {divide && <g className="divider"><line x1={sx(splitX)} y1={PAD} x2={sx(splitX)} y2={H-PAD}/><text x={sx(splitX)+10} y={PAD+20}>midX = 8.5</text></g>}
    <line className="axis" x1={PAD} y1={H-PAD} x2={W-PAD} y2={H-PAD}/><line className="axis" x1={PAD} y1={PAD} x2={PAD} y2={H-PAD}/>
    {stage==='distance' && <g className="distance-guide"><line x1={sx(2)} y1={sy(3)} x2={sx(8)} y2={sy(3)}/><line x1={sx(8)} y1={sy(3)} x2={sx(8)} y2={sy(7)}/><text x={(sx(2)+sx(8))/2} y={sy(3)+20}>Δx = 6</text><text x={sx(8)+8} y={(sy(3)+sy(7))/2}>Δy = 4</text><Pair ids={['A','C']} label="7.21"/></g>}
    {showL && <Pair ids={['A','C']} label="7.21"/>}{showR && <Pair ids={['D','E']} label="9.85"/>}{final && <Pair ids={['C','D']} primary label="1.41"/>}
    {P.map(p=>{
      const faded=(stage==='left'&&p.x>splitX)||(stage==='right'&&p.x<splitX)
      const hot=final&&['C','D'].includes(p.id)
      const rank=(stage==='stripY'||stage==='neighbors')?yOrder.findIndex(q=>q.id===p.id)+1:null
      return <g key={p.id} className={'pt '+(faded?'fade ':'')+(hot?'hot':'')} transform={'translate('+sx(p.x)+','+sy(p.y)+')'}>
        <circle r={hot?10:8}/><text className="pid" x="13" y="-10">{p.id}</text><text className="coord" x="13" y="8">({p.x},{p.y})</text>
        {rank && <text className="rank" x="-17" y="-12">{rank}</text>}
      </g>
    })}
    {stage==='sorted' && <g className="orders"><text x="90" y="31">Theo x: A → B → C → D → E → F</text><text x="90" y={H-18}>Theo y: A → F → C → D → E → B</text></g>}
    {stage==='neighbors' && <text className="neighbor-note" x={W-270} y={PAD+24}>7 là hằng số → quét strip vẫn O(n)</text>}
  </svg>
}

function Visual({stage}){
  if(stage==='concept') return <DacFlow/>
  if(stage==='complexity') return <ComplexityTree/>
  return <Plot stage={stage}/>
}

function App() {
  const [i,setI]=useState(0), [auto,setAuto]=useState(false), [cue,setCue]=useState(true)
  const s=steps[i], last=steps.length-1
  const move=d=>{setAuto(false);setI(v=>Math.min(last,Math.max(0,v+d)))}
  const fullscreen=()=>document.fullscreenElement?document.exitFullscreen?.():document.documentElement.requestFullscreen?.()

  useEffect(()=>{
    if(!auto) return
    if(i===last){setAuto(false);return}
    const t=setTimeout(()=>setI(v=>Math.min(last,v+1)),7000)
    return()=>clearTimeout(t)
  },[auto,i,last])

  useEffect(()=>{
    const key=e=>{
      if(e.key==='ArrowRight')move(1)
      else if(e.key==='ArrowLeft')move(-1)
      else if(e.key===' '){e.preventDefault();setAuto(v=>!v)}
      else if(e.key.toLowerCase()==='f')fullscreen()
      else if(e.key.toLowerCase()==='n')setCue(v=>!v)
      else if(e.key.toLowerCase()==='r'){setAuto(false);setI(0)}
    }
    addEventListener('keydown',key); return()=>removeEventListener('keydown',key)
  },[])

  return <main className="shell">
    <header>
      <div><p className="eyebrow">Từ số 0 → hiểu sâu → đủ để thuyết trình</p><h1>Closest Pair of Points</h1></div>
      <div className="head-actions"><button onClick={()=>setCue(v=>!v)}>{cue?'Ẩn lời thoại':'Hiện lời thoại'}</button><button onClick={fullscreen}>Toàn màn hình</button></div>
    </header>
    <div className="progress"><i style={{width:((i+1)/steps.length*100)+'%'}}/></div>
    <section className="layout">
      <div className="visual">
        <div className="visual-title"><div><span>{s.section} · Bước {i+1}/{steps.length}</span><h2>{s.title}</h2></div><p>{s.subtitle}</p></div>
        <div className="plotbox"><Visual stage={s.stage}/></div>
        <nav className="rail" aria-label="Các bước">{steps.map((x,n)=><button key={x.title} className={n===i?'active':''} onClick={()=>{setAuto(false);setI(n)}} aria-label={'Bước '+(n+1)+': '+x.title}>{n+1}</button>)}</nav>
      </div>
      <aside>
        <section><h3>Điều đang xảy ra</h3><ul>{s.bullets.map(x=><li key={x}>{x}</li>)}</ul></section>
        <section className="zero"><small>Hiểu từ số 0</small><p>{s.explain}</p></section>
        <section className="formula"><small>Công thức / kết luận</small><strong>{s.formula}</strong></section>
        <section className="why"><small>Tại sao bước này cần thiết?</small><p>{s.why}</p></section>
        <section><div className="code-head"><h3>Mã giả</h3><small>Dòng liên quan</small></div><pre>{pseudo.map((x,n)=><code key={x} className={n===s.line?'on':''}><b>{String(n+1).padStart(2,'0')}</b>{x}</code>)}</pre></section>
        <details className="check"><summary>Tự kiểm tra: {s.check[0]}</summary><p>{s.check[1]}</p></details>
        {cue && <section className="cue"><small>Bạn nói khi thuyết trình</small><p>{s.talk}</p></section>}
      </aside>
    </section>
    <footer><button disabled={i===0} onClick={()=>move(-1)}>← Bước trước</button><span>← → chuyển bước · Space tự chạy 7s · F toàn màn hình · N lời thoại · R về đầu</span><button className="auto" onClick={()=>setAuto(v=>!v)}>{auto?'Dừng':'Tự chạy'}</button><button className="next" disabled={i===last} onClick={()=>move(1)}>Bước tiếp →</button></footer>
  </main>
}

ReactDOM.createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>)
