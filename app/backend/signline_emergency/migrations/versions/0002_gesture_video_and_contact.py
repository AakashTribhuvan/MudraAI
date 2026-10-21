"""custom gestures: demonstration video, recording consent, emergency contact

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-20
"""

import sqlalchemy as sa
from alembic import op

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

COLUMNS = [
    sa.Column('video_storage_key', sa.String(length=80), nullable=True),
    sa.Column('video_mime_type', sa.String(length=40), nullable=True),
    sa.Column('video_duration_seconds', sa.Float(), nullable=True),
    sa.Column('video_size_bytes', sa.Integer(), nullable=True),
    sa.Column('recording_consent', sa.Boolean(), nullable=False, server_default=sa.false()),
    sa.Column('recording_consent_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('emergency_contact_id', sa.String(length=40), nullable=True),
]


def upgrade() -> None:
    with op.batch_alter_table('custom_gestures', schema=None) as batch_op:
        for column in COLUMNS:
            batch_op.add_column(column)


def downgrade() -> None:
    with op.batch_alter_table('custom_gestures', schema=None) as batch_op:
        for column in reversed(COLUMNS):
            batch_op.drop_column(column.name)
