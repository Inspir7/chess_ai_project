from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import chess
from PySide6.QtCore import Qt, QPoint, QTimer, QRectF, QSize
# ВАЖНО: QPen е тук, нужен за линиите на анализа
from PySide6.QtGui import QColor, QFont, QPainter, QRadialGradient, QPixmap, QImage, QPen
from PySide6.QtWidgets import QWidget
from PySide6.QtSvg import QSvgRenderer

# Забележка: GameController се импортира само за type hinting,
# но реалната инстанция идва отвън (от app.py)
from gui.game_controller import GameController
# Импорт за type hinting на анализа
from gui.analysis_controller import AnalysisController


class PieceRenderMode(str, Enum):
    UNICODE = "unicode"
    IMAGE = "image"


UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

PIECE_IMAGE_MAP = {
    "P": "wp.svg", "N": "wn.svg", "B": "wb.svg",
    "R": "wr.svg", "Q": "wq.svg", "K": "wk.svg",
    "p": "bp.svg", "n": "bn.svg", "b": "bb.svg",
    "r": "br.svg", "q": "bq.svg", "k": "bk.svg",
}

PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


class ChessBoardWidget(QWidget):
    def __init__(self, controller: GameController, analysis_controller: AnalysisController):
        super().__init__()

        # ВАЖНО: Запазваме контролерите, подадени отвън
        self.controller = controller
        self.analysis = analysis_controller

        # Правим alias (прякор) към дъската в контролера, за да не чупим rest of code
        self.chess_board = self.controller.board

        # Свързваме се със сигнала: когато дъската се промени -> прерисувай
        self.controller.board_changed.connect(self._on_board_changed)

        # Свързваме се със сигналите на анализа
        self.analysis.analysis_updated.connect(self.update)
        self.analysis.highlight_move.connect(self._on_highlight_move)

        self.setMinimumSize(420, 420)
        self.setMouseTracking(True)

        # --- Board colors ---
        self.light_color = QColor(242, 214, 222)
        self.dark_color = QColor(220, 132, 156)

        # --- Overlay Colors ---
        self.move_hint_color = QColor(255, 255, 255, 140)
        self.last_move_color = QColor(255, 230, 180, 110)
        self.check_glow_color = QColor(255, 60, 100)

        # --- Analysis Colors ---
        self.heatmap_base_color = QColor(255, 105, 180)
        self.hover_highlight_color = QColor(255, 215, 0, 180)  # Златисто

        # --- Piece Effects (Frames/Shadows) ---
        shadow_color = QColor(20, 20, 20, 110)

        self.white_piece_frame_color = shadow_color
        self.black_piece_frame_color = shadow_color

        self.white_outline_scale = 1.05
        self.white_outline_opacity = 0.6
        self.black_outline_scale = 1.05
        self.black_outline_opacity = 0.6

        # --- Selection Glow ---
        self.white_selection_glow = QColor(255, 255, 255, 180)
        self.black_selection_glow = QColor(140, 60, 90, 160)

        # --- Animation State ---
        self.pulse_value = 0.0
        self.pulse_direction = 1
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._update_pulse)
        self.anim_timer.start(50)

        # --- Selection State ---
        self.selected_square: Optional[int] = None
        self.legal_moves: List[chess.Move] = []

        # --- Promotion State ---
        self.pending_promotion_move: Optional[chess.Move] = None
        self.hovered_promotion_index: Optional[int] = None
        self.pressed_promotion_index: Optional[int] = None
        self.promotion_icons_size_ratio = 0.62

        # --- Hover State ---
        self.current_hover_move = None

        # --- Rendering Assets ---
        self.render_mode = PieceRenderMode.IMAGE
        self.assets_dir = Path(__file__).resolve().parent / "assets" / "barbie"
        self.piece_renderers: Dict[str, QSvgRenderer] = {}

        self._silhouette_cache: Dict[Tuple[str, int, int, str], QPixmap] = {}
        self._last_rendered_s = 0

        self._load_piece_images()

    # ---------------- Logic Integration ----------------

    def _on_board_changed(self):
        """Извиква се от контролера, когато някой (човек или AI) направи ход."""
        self.update()  # Веднага обновяваме екрана

    def _on_highlight_move(self, move):
        """Вика се, когато минем с мишката върху ход в списъка."""
        self.current_hover_move = move
        self.update()

    # ---------------- Animation Loop ----------------

    def _update_pulse(self):
        step = 0.05
        if self.pulse_direction == 1:
            self.pulse_value += step
            if self.pulse_value >= 1.0:
                self.pulse_value = 1.0
                self.pulse_direction = -1
        else:
            self.pulse_value -= step
            if self.pulse_value <= 0.0:
                self.pulse_value = 0.0
                self.pulse_direction = 1

        if self.chess_board.is_check():
            self.update()

    # ---------------- Public API ----------------

    def set_render_mode(self, mode: PieceRenderMode):
        self.render_mode = mode
        self.update()

    # ---------------- Events ----------------

    def mouseMoveEvent(self, event):
        if self.pending_promotion_move is None:
            self.setCursor(Qt.ArrowCursor)
            if self.hovered_promotion_index is not None:
                self.hovered_promotion_index = None
                self.update()
            return

        idx = self._promotion_index_from_pos(event.position().toPoint())
        if idx != self.hovered_promotion_index:
            self.hovered_promotion_index = idx
            self.update()
        self.setCursor(Qt.PointingHandCursor if idx is not None else Qt.ArrowCursor)

    def leaveEvent(self, event):
        self.hovered_promotion_index = None
        self.setCursor(Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        pos = event.position().toPoint()

        # 1. Логика за промоция (ако менюто е отворено)
        if self.pending_promotion_move:
            idx = self._promotion_index_from_pos(pos)
            choice = self._promotion_choice_from_click(pos)
            if choice is not None and idx is not None:
                self.pressed_promotion_index = idx
                self.update()

                def apply():
                    base = self.pending_promotion_move
                    if base:
                        # Създаваме пълния ход с промоция
                        move = chess.Move(base.from_square, base.to_square, promotion=choice)
                        # Изпращаме към контролера!
                        self.controller.on_human_move(move)

                    self.pending_promotion_move = None
                    self.hovered_promotion_index = None
                    self.pressed_promotion_index = None
                    self.update()

                QTimer.singleShot(120, apply)
                return

            # Ако кликнем извън менюто, отказваме промоцията
            self.pending_promotion_move = None
            self.hovered_promotion_index = None
            self.pressed_promotion_index = None
            self.update()
            return  # Важно: спираме тук

        # 2. Логика за местене по дъската
        square = self._square_from_pos(pos)
        if square is None:
            self._clear_selection()
            self.update()
            return

        # Ако кликнем върху валиден ход за избраната фигура
        if self.selected_square is not None:
            for move in self.legal_moves:
                if move.to_square == square:
                    # Ако е промоция, не местим веднага, а отваряме менюто
                    if move.promotion:
                        self.pending_promotion_move = move
                        self._clear_selection()
                        self.update()
                        return

                    # НОРМАЛЕН ХОД: Изпращаме към контролера
                    self.controller.on_human_move(move)

                    self._clear_selection()
                    self.update()
                    return

        # 3. Логика за селектиране на фигура
        piece = self.chess_board.piece_at(square)
        # Позволяваме селекция само ако е ред на човека (по желание може и винаги)
        # Тук използваме board.turn, за да видим чий ред е
        if piece and piece.color == self.chess_board.turn:
            self.selected_square = square
            self.legal_moves = [
                m for m in self.chess_board.legal_moves
                if m.from_square == square
            ]
        else:
            self._clear_selection()

        self.update()

    def _clear_selection(self):
        self.selected_square = None
        self.legal_moves = []

    # ---------------- Painting ----------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        size = min(self.width(), self.height())
        s = size // 8
        x0 = (self.width() - size) // 2
        y0 = (self.height() - size) // 2

        if s != self._last_rendered_s:
            self._silhouette_cache.clear()
            self._last_rendered_s = s

        self._draw_squares(painter, x0, y0, s)
        self._draw_last_move(painter, x0, y0, s)
        self._draw_check_highlight(painter, x0, y0, s)
        self._draw_selection_glow(painter, x0, y0, s)

        # Heatmap
        # if self.analysis.active:
        #    self._draw_heatmap(painter, x0, y0, s)

        # Analysis Hover Highlight (рисува се преди фигурите)
        if self.current_hover_move:
            self._draw_hover_highlight(painter, x0, y0, s)

        self._draw_move_hints(painter, x0, y0, s)
        self._draw_pieces(painter, x0, y0, s)

        if self.pending_promotion_move:
            self._draw_promotion_menu(painter, x0, y0, s)

    # ---------------- Board Drawing Helpers ----------------

    def _draw_squares(self, painter, x0, y0, s):
        for r in range(8):
            for c in range(8):
                painter.fillRect(x0 + c * s, y0 + r * s, s, s,
                                 self.light_color if (r + c) % 2 == 0 else self.dark_color)

    def _draw_last_move(self, painter, x0, y0, s):
        if not self.chess_board.move_stack:
            return
        move = self.chess_board.peek()
        for sq in [move.from_square, move.to_square]:
            col = chess.square_file(sq)
            row = 7 - chess.square_rank(sq)
            rect = QRectF(x0 + col * s, y0 + row * s, s, s)
            painter.setBrush(self.last_move_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 4, 4)

    def _draw_selection_glow(self, painter, x0, y0, s):
        if self.selected_square is None:
            return
        col = chess.square_file(self.selected_square)
        row = 7 - chess.square_rank(self.selected_square)
        piece = self.chess_board.piece_at(self.selected_square)

        base_color = self.white_selection_glow if piece and piece.color == chess.WHITE else self.black_selection_glow

        center = QPoint(x0 + col * s + s // 2, y0 + row * s + s // 2)
        radius = s * 0.55
        gradient = QRadialGradient(center, radius)
        gradient.setColorAt(0.0, base_color)
        gradient.setColorAt(1.0, QColor(base_color.red(), base_color.green(), base_color.blue(), 0))

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, int(radius), int(radius))

    def _draw_check_highlight(self, painter, x0, y0, s):
        if not self.chess_board.is_check():
            return
        king_sq = self.chess_board.king(self.chess_board.turn)
        if king_sq is None:
            return
        col = chess.square_file(king_sq)
        row = 7 - chess.square_rank(king_sq)

        cx = x0 + col * s + s // 2
        cy = y0 + row * s + s // 2

        base_r = s * 0.45
        pulse_add = (s * 0.12) * self.pulse_value
        current_radius = base_r + pulse_add

        alpha = 80 + int(90 * self.pulse_value)
        color = QColor(self.check_glow_color)
        color.setAlpha(alpha)

        gradient = QRadialGradient(cx, cy, current_radius)
        gradient.setColorAt(0.0, color)
        gradient.setColorAt(0.7, color)
        gradient.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPoint(cx, cy), int(current_radius), int(current_radius))

    def _draw_move_hints(self, painter, x0, y0, s):
        painter.setBrush(self.move_hint_color)
        painter.setPen(Qt.NoPen)
        for m in self.legal_moves:
            col = chess.square_file(m.to_square)
            row = 7 - chess.square_rank(m.to_square)
            painter.drawEllipse(
                QPoint(x0 + col * s + s // 2, y0 + row * s + s // 2),
                int(s * 0.12), int(s * 0.12)
            )

    # ---------------- Analysis Drawing ----------------

    def _draw_heatmap(self, painter, x0, y0, s):
        policy = self.analysis.current_policy
        if not policy: return
        max_prob = max(policy.values()) if policy else 1.0
        for move, prob in policy.items():
            sq = move.to_square
            col = chess.square_file(sq)
            row = 7 - chess.square_rank(sq)
            alpha = int(30 + (prob / max_prob) * 150)
            color = QColor(self.heatmap_base_color)
            color.setAlpha(alpha)
            rect = QRectF(x0 + col * s, y0 + row * s, s, s)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect.adjusted(4, 4, -4, -4), 8, 8)

    def _draw_hover_highlight(self, painter, x0, y0, s):
        move = self.current_hover_move
        # Рисуваме From и To
        self._draw_single_highlight(painter, x0, y0, s, move.from_square)
        self._draw_single_highlight(painter, x0, y0, s, move.to_square)

        # Линия
        start_sq = move.from_square
        end_sq = move.to_square
        c1 = chess.square_file(start_sq);
        r1 = 7 - chess.square_rank(start_sq)
        c2 = chess.square_file(end_sq);
        r2 = 7 - chess.square_rank(end_sq)
        p1 = QPoint(x0 + c1 * s + s // 2, y0 + r1 * s + s // 2)
        p2 = QPoint(x0 + c2 * s + s // 2, y0 + r2 * s + s // 2)

        pen = QPen(self.hover_highlight_color)
        pen.setWidth(4)
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

    def _draw_single_highlight(self, painter, x0, y0, s, sq):
        col = chess.square_file(sq)
        row = 7 - chess.square_rank(sq)
        rect = QRectF(x0 + col * s, y0 + row * s, s, s)

        # Рамка
        painter.setBrush(Qt.NoBrush)
        pen = QPen(self.hover_highlight_color)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 5, 5)

        # Фон
        color = QColor(self.hover_highlight_color)
        color.setAlpha(50)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 5, 5)

    # ---------------- Piece Rendering ----------------

    def _load_piece_images(self):
        if self.piece_renderers:
            return
        for sym, fn in PIECE_IMAGE_MAP.items():
            path = self.assets_dir / fn
            if path.exists():
                self.piece_renderers[sym] = QSvgRenderer(str(path))

    def _draw_pieces(self, painter, x0, y0, s):
        for sq, piece in self.chess_board.piece_map().items():
            col = chess.square_file(sq)
            row = 7 - chess.square_rank(sq)
            if self.render_mode == PieceRenderMode.IMAGE:
                self._draw_svg_piece(painter, piece, x0, y0, s, row, col)
            else:
                self._draw_unicode_piece(painter, piece, x0, y0, s, row, col)

    def _generate_silhouette(self, renderer: QSvgRenderer, w: int, h: int, color: QColor, piece_sym: str) -> QPixmap:
        key = (piece_sym, w, h, color.name())
        if key in self._silhouette_cache:
            return self._silhouette_cache[key]

        img = QImage(w, h, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        renderer.render(p)

        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(img.rect(), color)
        p.end()

        pix = QPixmap.fromImage(img)
        self._silhouette_cache[key] = pix
        return pix

    def _rect_for_svg_in_cell(self, renderer: QSvgRenderer, x0: int, y0: int, s: int, r: int, c: int) -> Optional[
        QRectF]:
        pad = s * 0.15
        max_s = s - 2 * pad

        svg_size = renderer.viewBox().size() if not renderer.viewBox().isNull() else renderer.defaultSize()
        if svg_size.isEmpty() or svg_size.height() == 0:
            return None

        ar = svg_size.width() / svg_size.height()

        draw_h = max_s
        draw_w = draw_h * ar
        if draw_w > max_s:
            draw_w = max_s
            draw_h = draw_w / ar if ar != 0 else max_s

        return QRectF(
            x0 + c * s + (s - draw_w) / 2,
            y0 + r * s + (s - draw_h) / 2,
            draw_w,
            draw_h
        )

    def _scaled_rect_centered(self, rect: QRectF, scale: float) -> QRectF:
        return QRectF(
            rect.center().x() - rect.width() * scale / 2,
            rect.center().y() - rect.height() * scale / 2,
            rect.width() * scale,
            rect.height() * scale
        )

    def _draw_svg_piece(self, painter, piece, x0, y0, s, r, c):
        renderer = self.piece_renderers.get(piece.symbol())
        if not renderer:
            return

        rect = self._rect_for_svg_in_cell(renderer, x0, y0, s, r, c)
        if rect is None:
            return

        is_white = piece.color == chess.WHITE

        # --- РАМКА / СЯНКА ---
        if is_white:
            color = self.white_piece_frame_color
            scale = self.white_outline_scale
            opacity = self.white_outline_opacity
        else:
            color = self.black_piece_frame_color
            scale = self.black_outline_scale
            opacity = self.black_outline_opacity

        offset = min(3.0, max(1.0, s * 0.03))

        silhouette_rect = self._scaled_rect_centered(rect, scale)
        shadow_w = int(silhouette_rect.width())
        shadow_h = int(silhouette_rect.height())

        if shadow_w > 0 and shadow_h > 0:
            shadow_pix = self._generate_silhouette(renderer, shadow_w, shadow_h, color, piece.symbol())

            painter.save()
            painter.setOpacity(opacity)
            painter.drawPixmap(
                int(silhouette_rect.x() + offset),
                int(silhouette_rect.y() + offset),
                shadow_pix
            )
            painter.restore()

        # --- САМАТА ФИГУРА ---
        painter.setOpacity(1.0 if is_white else 0.95)
        renderer.render(painter, rect)
        painter.setOpacity(1.0)

    def _draw_unicode_piece(self, painter, piece, x0, y0, s, r, c):
        char = UNICODE_PIECES.get(piece.symbol())
        if not char:
            return
        font = QFont("DejaVu Sans", int(s * 0.56))
        font.setWeight(QFont.Medium)
        painter.setFont(font)
        painter.setPen(QColor(40, 40, 40))
        painter.drawText(QRectF(x0 + c * s, y0 + r * s, s, s), Qt.AlignCenter, char)

    # ---------------- Promotion Menu ----------------

    def _draw_promotion_menu(self, painter, x0, y0, s):
        mx, mt, mw, mh = self._promotion_menu_geometry(x0, y0, s)
        painter.setBrush(QColor(180, 120, 140, 70))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(mx + 3, mt + 4, mw, mh, 14, 14)
        painter.setBrush(QColor(255, 245, 248, 240))
        painter.setPen(QColor(215, 150, 165, 180))
        painter.drawRoundedRect(mx, mt, mw, mh, 14, 14)

        for i, promo in enumerate(PROMOTION_PIECES):
            cell_y = mt + i * s
            if self.pressed_promotion_index == i:
                bg = QColor(255, 210, 230, 250)
                scale = self.promotion_icons_size_ratio * 0.97
            elif self.hovered_promotion_index == i:
                bg = QColor(255, 235, 245, 245)
                scale = self.promotion_icons_size_ratio * 1.05
            else:
                bg = QColor(255, 255, 255, 215)
                scale = self.promotion_icons_size_ratio

            painter.setBrush(bg)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(mx + 6, cell_y + 6, mw - 12, s - 12, 10, 10)
            self._draw_promotion_image_scaled(painter, promo, mx + mw // 2, cell_y + s // 2, s, scale)

    def _draw_promotion_image_scaled(self, painter, promo, cx, cy, s, scale):
        piece = chess.Piece(promo, self.chess_board.turn)
        renderer = self.piece_renderers.get(piece.symbol())
        if not renderer: return
        svg_size = renderer.viewBox().size() if not renderer.viewBox().isNull() else renderer.defaultSize()
        ar = svg_size.width() / svg_size.height()
        draw_h = s * scale
        draw_w = draw_h * ar
        rect = QRectF(cx - draw_w / 2, cy - draw_h / 2, draw_w, draw_h)

        is_white_turn = self.chess_board.turn == chess.WHITE
        color = self.white_piece_frame_color if is_white_turn else self.black_piece_frame_color
        outline_scale = self.white_outline_scale if is_white_turn else self.black_outline_scale
        outline_opacity = self.white_outline_opacity if is_white_turn else self.black_outline_opacity

        silhouette_rect = self._scaled_rect_centered(rect, outline_scale)
        shadow_w = int(silhouette_rect.width())
        shadow_h = int(silhouette_rect.height())

        if shadow_w > 0 and shadow_h > 0:
            shadow_pix = self._generate_silhouette(renderer, shadow_w, shadow_h, color, piece.symbol())
            offset = s * 0.03
            painter.save()
            painter.setOpacity(outline_opacity)
            painter.drawPixmap(int(silhouette_rect.x() + offset), int(silhouette_rect.y() + offset), shadow_pix)
            painter.restore()

        renderer.render(painter, rect)

    # ---------------- Utils ----------------

    def _promotion_menu_geometry(self, x0, y0, s):
        to_sq = self.pending_promotion_move.to_square
        col = chess.square_file(to_sq)
        row = 7 - chess.square_rank(to_sq)
        mx = x0 + col * s
        base_y = y0 + row * s
        mh = s * len(PROMOTION_PIECES)
        mt = base_y - mh if base_y - mh >= y0 else base_y + s
        return mx, mt, s, mh

    def _promotion_index_from_pos(self, pos):
        if not self.pending_promotion_move: return None
        size = min(self.width(), self.height())
        s = size // 8
        x0 = (self.width() - size) // 2
        y0 = (self.height() - size) // 2
        mx, mt, mw, mh = self._promotion_menu_geometry(x0, y0, s)
        if mx <= pos.x() <= mx + mw and mt <= pos.y() <= mt + mh:
            idx = int((pos.y() - mt) // s)
            if 0 <= idx < len(PROMOTION_PIECES): return idx
        return None

    def _promotion_choice_from_click(self, pos):
        idx = self._promotion_index_from_pos(pos)
        return PROMOTION_PIECES[idx] if idx is not None else None

    def _maybe_play_ai(self):
        if self.controller.is_ai_turn() and not self.controller.board.is_game_over():
            # Използваме таймера, който вече имаш, за да се спази ai_delay
            QTimer.singleShot(self.controller.ai_delay, self.controller.play_ai_move)

    def _square_from_pos(self, pos):
        size = min(self.width(), self.height())
        s = size // 8
        x0 = (self.width() - size) // 2
        y0 = (self.height() - size) // 2
        x, y = pos.x() - x0, pos.y() - y0
        if x < 0 or y < 0: return None
        col = x // s
        row = y // s
        if 0 <= col <= 7 and 0 <= row <= 7:
            return chess.square(col, 7 - row)
        return None