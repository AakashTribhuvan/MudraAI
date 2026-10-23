"""incidents: gesture report link, consented video copy and sharing status

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-21
"""

import sqlalchemy as sa
from alembic import op

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

COLUMNS = [
    sa.Column('gesture_id', sa.String(length=40), nullable=True),
    sa.Column('emergency_contact_id', sa.String(length=40), nullable=True),
    sa.Column('emergency_message', sa.Text(), nullable=True),
    sa.Column('video_storage_key', sa.String(length=80), nullable=True),
    sa.Column('video_mime_type', sa.String(length=40), nullable=True),
    sa.Column('video_duration_seconds', sa.Float(), nullable=True),
    sa.Column('sharing_consent', sa.Boolean(), nullable=False, server_default=sa.false()),
    sa.Column('sharing_consent_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('video_sharing_status', sa.String(length=20), nullable=False, server_default='NOT_ATTACHED'),
]


def upgrade() -> None:
    with op.batch_alter_table('emergency_incidents', schema=None) as batch_op:
        for column in COLUMNS:
            batch_op.add_column(column)


def downgrade() -> None:
    with op.batch_alter_table('emergency_incidents', schema=None) as batch_op:
        for column in reversed(COLUMNS):
            batch_op.drop_column(column.name)
