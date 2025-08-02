"""Initial migration - Create all tables

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_superuser', sa.Boolean(), nullable=False),
        sa.Column('avatar_url', sa.String(length=500), nullable=True),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('department', sa.String(length=100), nullable=True),
        sa.Column('job_title', sa.String(length=100), nullable=True),
        sa.Column('email_verified', sa.Boolean(), nullable=False),
        sa.Column('email_verified_at', sa.DateTime(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False),
        sa.Column('locked_until', sa.DateTime(), nullable=True),
        sa.Column('preferences', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('api_key', sa.String(length=255), nullable=True),
        sa.Column('api_key_created_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('created_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('metadata_', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('api_key')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)
    op.create_index(op.f('ix_users_created_at'), 'users', ['created_at'], unique=False)
    op.create_index(op.f('ix_users_updated_at'), 'users', ['updated_at'], unique=False)
    op.create_index(op.f('ix_users_deleted_at'), 'users', ['deleted_at'], unique=False)
    
    # Create workflows table
    op.create_table('workflows',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('status', sa.Enum('DRAFT', 'ACTIVE', 'INACTIVE', 'ARCHIVED', name='workflowstatus'), nullable=False),
        sa.Column('definition', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('input_schema', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('output_schema', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('triggers', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('timeout_seconds', sa.Integer(), nullable=False),
        sa.Column('max_retries', sa.Integer(), nullable=False),
        sa.Column('retry_delay_seconds', sa.Integer(), nullable=False),
        sa.Column('execution_count', sa.Integer(), nullable=False),
        sa.Column('success_count', sa.Integer(), nullable=False),
        sa.Column('failure_count', sa.Integer(), nullable=False),
        sa.Column('average_duration', sa.Float(), nullable=True),
        sa.Column('created_by_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('created_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('metadata_', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(['created_by_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflows_id'), 'workflows', ['id'], unique=False)
    op.create_index(op.f('ix_workflows_created_by_id'), 'workflows', ['created_by_id'], unique=False)
    op.create_index(op.f('ix_workflows_created_at'), 'workflows', ['created_at'], unique=False)
    op.create_index(op.f('ix_workflows_updated_at'), 'workflows', ['updated_at'], unique=False)
    op.create_index(op.f('ix_workflows_deleted_at'), 'workflows', ['deleted_at'], unique=False)
    
    # Create documents table
    op.create_table('documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('mime_type', sa.String(length=100), nullable=False),
        sa.Column('checksum', sa.String(length=64), nullable=False),
        sa.Column('document_type', sa.Enum('INVOICE', 'RECEIPT', 'CONTRACT', 'REPORT', 'EMAIL', 'FORM', 'IMAGE', 'OTHER', name='documenttype'), nullable=False),
        sa.Column('status', sa.Enum('UPLOADED', 'PROCESSING', 'PROCESSED', 'FAILED', 'ARCHIVED', name='documentstatus'), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('title', sa.String(length=500), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('extracted_text', sa.Text(), nullable=True),
        sa.Column('processing_started_at', sa.DateTime(), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(), nullable=True),
        sa.Column('processing_duration', sa.Float(), nullable=True),
        sa.Column('processing_steps', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('ocr_text', sa.Text(), nullable=True),
        sa.Column('ocr_confidence', sa.Float(), nullable=True),
        sa.Column('ocr_language', sa.String(length=10), nullable=True),
        sa.Column('entities', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('keywords', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('sentiment', sa.String(length=20), nullable=True),
        sa.Column('extracted_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('validation_results', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('owner_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('search_vector', sa.Text(), nullable=True),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('created_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('metadata_', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_documents_id'), 'documents', ['id'], unique=False)
    op.create_index(op.f('ix_documents_checksum'), 'documents', ['checksum'], unique=False)
    op.create_index(op.f('ix_documents_owner_id'), 'documents', ['owner_id'], unique=False)
    op.create_index(op.f('ix_documents_created_at'), 'documents', ['created_at'], unique=False)
    op.create_index(op.f('ix_documents_updated_at'), 'documents', ['updated_at'], unique=False)
    op.create_index(op.f('ix_documents_deleted_at'), 'documents', ['deleted_at'], unique=False)
    
    # Create workflow_executions table
    op.create_table('workflow_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('execution_id', sa.String(length=50), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', name='executionstatus'), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('input_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('output_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('current_step', sa.Integer(), nullable=False),
        sa.Column('total_steps', sa.Integer(), nullable=False),
        sa.Column('steps', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('timeout_at', sa.DateTime(), nullable=True),
        sa.Column('context', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('created_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('updated_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['workflow_id'], ['workflows.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('execution_id')
    )
    op.create_index(op.f('ix_workflow_executions_id'), 'workflow_executions', ['id'], unique=False)
    op.create_index(op.f('ix_workflow_executions_execution_id'), 'workflow_executions', ['execution_id'], unique=False)
    op.create_index(op.f('ix_workflow_executions_workflow_id'), 'workflow_executions', ['workflow_id'], unique=False)
    op.create_index(op.f('ix_workflow_executions_user_id'), 'workflow_executions', ['user_id'], unique=False)
    op.create_index(op.f('ix_workflow_executions_document_id'), 'workflow_executions', ['document_id'], unique=False)
    op.create_index(op.f('ix_workflow_executions_created_at'), 'workflow_executions', ['created_at'], unique=False)
    op.create_index(op.f('ix_workflow_executions_updated_at'), 'workflow_executions', ['updated_at'], unique=False)
    op.create_index(op.f('ix_workflow_executions_deleted_at'), 'workflow_executions', ['deleted_at'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('workflow_executions')
    op.drop_table('documents')
    op.drop_table('workflows')
    op.drop_table('users')
    
    # Drop custom enum types
    op.execute('DROP TYPE IF EXISTS executionstatus')
    op.execute('DROP TYPE IF EXISTS documentstatus')
    op.execute('DROP TYPE IF EXISTS documenttype')
    op.execute('DROP TYPE IF EXISTS workflowstatus')