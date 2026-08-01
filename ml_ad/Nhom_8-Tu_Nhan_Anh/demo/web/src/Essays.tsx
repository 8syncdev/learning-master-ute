import { Lightbulb, GitCompare, Rocket, Cpu, Brain, Eye, Scale, Zap, TrendingUp } from "lucide-react";

/* ============================================================
   Essays — 3 khối giáo dục tĩnh:
   1. PaperIdeal  — người mới hiểu được ideal của paper
   2. WhyFRF      — vì sao chọn FRF-MLP dù softmax/m-BERT điểm cao hơn
   3. FutureWork  — hướng cải tiến (ONNX, ANFIS, PhoBERT ...)
   ============================================================ */

export function PaperIdeal() {
  return (
    <div className="essay">
      <h3><Lightbulb size={16} /> Ý tưởng chính của paper</h3>
      <div className="e-grid">
        <div className="e-card">
          <div className="e-h">Vấn đề</div>
          <p>Bình luận công kích và thù ghét lan truyền nhanh trên mạng xã hội Việt Nam. Cần bộ phát hiện tự động, nhưng ranh giới giữa CLEAN / OFFENSIVE / HATE vốn <em>mờ</em> — hệ số đồng thuận kappa giữa người gán nhãn chỉ 0,52.</p>
        </div>
        <div className="e-card">
          <div className="e-h">Hai cực lưỡng lẹn</div>
          <p>Thuần mạng nơ-ron (MLP, BERT): chính xác nhưng <em>hộp đen</em>, nặng, khó giải thích. Thuần luật/từ điển: diễn giải được nhưng <em>thiếu độ phủ</em> trước teencode biến đổi. Cả hai đều yếu ở biên OFFENSIVE/HATE.</p>
        </div>
        <div className="e-card hl">
          <div className="e-h">Giải pháp FRF-MLP</div>
          <p>Kết hợp <strong>cả hai</strong>: MLP học thống kê từ TF-IDF, còn một <strong>hệ luật mờ Mamdani</strong> nắm bắt tri thức ngôn ngữ (mức tục tĩu, mật độ, độ nhắm đích). Hai kênh hợp nhất ở 2 cấp độ (đặc trưng + quyết định) — vừa chính xác vừa <strong>diễn giải được từng bước</strong>.</p>
        </div>
      </div>
      <div className="flow">
        <span>bình luận</span><span className="arrow">→</span> <span>Tiền xử lý</span><span className="arrow">→</span> <span>TF-IDF ⊕ Lexicon mờ</span><span className="arrow">→</span> <span>MLP + 7 luật Mamdani</span><span className="arrow">→</span> <span>fusion</span><span className="arrow">→</span> <span className="win">CLEAN / OFFENSIVE / HATE</span>
      </div>
    </div>
  );
}

export function WhyFRF() {
  return (
    <div className="essay">
      <h3><GitCompare size={16} /> Vì sao chọn FRF-MLP dù softmax / m-BERT điểm cao hơn?</h3>
      <p className="lede">
        Trên ViHSD, hồi quy softmax đạt <strong>64,50%</strong> macro-F1 và m-BERT đạt accuracy <strong>86,88%</strong> — đều nhỉnh hơn FRF-MLP (63,00% / 84,33%). Vẫn chọn FRF-MLP vì <strong>4 lý do cụ thể</strong>, đều liên quan đến tiêu chí ứng dụng thực tế chứ không chỉ số đo đơn lẻ:
      </p>
      <div className="why-grid">
        <Reason icon={Eye} title="Diễn giải được (explainability)"
          body="Khi một bình luận bị gán HATE, giáo viên / người kiểm duyệt cần biết VÌ SAO. FRF-MLP chỉ ra token nào công kích (z-score), luật Mamdani nào kích hoạt, và xác suất từ mỗi kênh. Softmax và m-BERT là hộp đen — không trả lời được câu hỏi đó." />
        <Reason icon={Cpu} title="Nhẹ + CPU + real-time"
          body="10,3 triệu tham số, 42 MB, suy diễn ~3 ms trên CPU thường — nhẹ hơn m-BERT khoảng 17× (178 triệu tham số, ~700 MB, thường cần GPU). Đủ chạy thời gian thực trong đường ống kiểm duyệt hoặc cạnh biên (edge)." />
        <Reason icon={Scale} title="Lớp thiểu số được nâng"
          body="Macro-F1 trọng số lớp thiểu số quan trọng hơn accuracy. FRF-MLP tăng recall OFFENSIVE (40,3→45,3%) và HATE (54,4→57,4%) so với MLP thuần — chính vùng khó mà hệ kiểm duyệt quan tâm nhất. Softmax tuy macro-F1 cao nhưng là mô hình tuyến tính thuần, không nắm được tương tác phi tuyến." />
        <Reason icon={Brain} title="Tri thức ngôn ngữ tường minh"
          body="Hệ mờ mã hoá trực tiếp khái niệm 'nhắm đích' (T) — chìa phân biệt HATE với OFFENSIVE theo định nghĩa nhãn ViHSD. Kiến thức này có thể chỉnh sửa / kiểm chứng độc lập với dữ liệu; m-BERT học ẩn trong embedding, không kiểm soát được." />
      </div>
      <div className="note">
        <strong>Trung thực:</strong> softmax class-weighted là baseline rất mạnh trên TF-IDF thưa (hiện tượng đã biết — mô hình tuyến tính lồi hội tụ toàn cục). FRF-MLP không bỏ qua điều đó; paper nêu thẳng và coi 'áp cơ chế fusion mờ lên softmax' là hướng mở. Lựa chọn FRF-MLP là đánh đổi có chủ đích: <em>chấp nhận giảm nhẹ một số đo để đổi lấy diễn giải + nhẹ + cải thiện lớp thiểu số</em>.
      </div>
    </div>
  );
}

export function FutureWork() {
  return (
    <div className="essay">
      <h3><Rocket size={16} /> Hướng cải tiến tiếp theo</h3>
      <div className="fw-list">
        <FW icon={Zap} title="Export ONNX / quantint8 — suy diễn nhanh hơn 5–10×"
          body="Đã có PyTorch weights → torch.onnx.export sang ONNX, rồi onnxruntime chạy trên CPU. Kết hợp quantization int8 đưa latency từ ~3 ms xuống ~0,3–0,5 ms, thông lượng vài nghìn bình luận/giây trên 1 core — đủ cho đường ống production lớn." />
        <FW icon={Brain} title="Học biên hàm thành viên + trọng số luật (ANFIS)"
          body="Hiện biên LOW/MED/HIGH và trọng số luật cố định thủ công. Cho gradient chảy qua hàm thành viên (differentiable fuzzy) và trọng số luật, huấn luyện đầu-cuối → hệ mờ tự tinh chỉnh theo dữ liệu thay vì靠 thiết kế tay." />
        <FW icon={TrendingUp} title="Đầu vào embedding PhoBERT thay TF-IDF"
          body="TF-IDF mất ngữ cảnh và thứ tự. Thay bằng PhoBERT (đã tiền huấn luyện tiếng Việt) làm bộ nhúng, giữ nguyên nhánh mờ + MLP phía trên → kỳ vọng macro-F1 vượt 70%, vẫn giữ khả năng diễn giải của kênh mờ." />
        <FW icon={GitCompare} title="Áp fusion mờ lên bộ học mạnh hơn"
          body="Cơ chế 2 cấp (feature + decision fusion) tổng quát, không phụ thuộc MLP. Áp lên softmax, LightGBM, hay chính m-BERT: nối đặc trưng mờ vào đầu vào + λ-fusion đầu ra — khả năng cải thiện macro-F1 cho mọi bộ phân loại, đặc biệt lớp thiểu số." />
        <FW icon={Scale} title="Mở rộng lexicon + teencode + đa dữ liệu"
          body="Lexicon log-odds hiện học từ ViHSD; bổ sung từ điển teencode tiếng Việt (Hatebase, Hurtlex) và dữ liệu VLSP-HSD để phủ teencode mới và ngôn ngữ biến đổi. Cộng sampling lớp thiểu số (SMOTE) hoặc focal loss để giảm thêm khoảng cách CLEAN/OFFENSIVE." />
      </div>
    </div>
  );
}

/* ---- small bits ---- */
function Reason({ icon: Icon, title, body }: { icon: typeof Eye; title: string; body: string }) {
  return (
    <div className="reason">
      <div className="r-icon"><Icon size={16} /></div>
      <div>
        <div className="r-title">{title}</div>
        <p>{body}</p>
      </div>
    </div>
  );
}
function FW({ icon: Icon, title, body }: { icon: typeof Zap; title: string; body: string }) {
  return (
    <div className="fw">
      <div className="fw-h"><Icon size={15} /> {title}</div>
      <p>{body}</p>
    </div>
  );
}
