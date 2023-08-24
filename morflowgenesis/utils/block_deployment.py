from uuid import UUID

from prefect.blocks.system import Secret
from prefect.client.orchestration import ServerType, get_client
from prefect.deployments import Deployment
from prefect.exceptions import ObjectAlreadyExists
from prefect.utilities.asyncutils import sync_compatible


class BlockDeployment(Deployment):
    @sync_compatible
    async def apply(
        self, pull_cfg=None, upload: bool = False, work_queue_concurrency: int = None
    ) -> UUID:
        """Registers this deployment with the API and returns the deployment's ID.

        Args:
            upload: if True, deployment files are automatically uploaded to remote
                storage
            work_queue_concurrency: If provided, sets the concurrency limit on the
                deployment's work queue
        """
        if not self.name or not self.flow_name:
            raise ValueError("Both a deployment name and flow name must be set.")
        async with get_client() as client:
            # prep IDs
            flow_id = await client.create_flow_from_name(self.flow_name)

            infrastructure_document_id = self.infrastructure._block_document_id
            if not infrastructure_document_id:
                # if not building off a block, will create an anonymous block
                self.infrastructure = self.infrastructure.copy()
                infrastructure_document_id = await self.infrastructure._save(
                    is_anonymous=True,
                )

            if upload:
                await self.upload_to_storage()

            if self.work_queue_name and work_queue_concurrency is not None:
                try:
                    res = await client.create_work_queue(
                        name=self.work_queue_name, work_pool_name=self.work_pool_name
                    )
                except ObjectAlreadyExists:
                    res = await client.read_work_queue_by_name(
                        name=self.work_queue_name, work_pool_name=self.work_pool_name
                    )
                await client.update_work_queue(res.id, concurrency_limit=work_queue_concurrency)

            pull_steps = []
            if pull_cfg is not None:
                # pre-create access token secret block
                access_token_secret_block = await Secret.load(pull_cfg["secret_block_name"])
                pull_steps.append(
                    {
                        "prefect.deployments.steps.git_clone": {
                            "repository": pull_cfg["repository"],
                            "branch": pull_cfg.get("branch", "main"),
                            "access_token": access_token_secret_block.get(),
                        }
                    }
                )

            # we assume storage was already saved
            storage_document_id = getattr(self.storage, "_block_document_id", None)
            deployment_id = await client.create_deployment(
                flow_id=flow_id,
                name=self.name,
                work_queue_name=self.work_queue_name,
                work_pool_name=self.work_pool_name,
                version=self.version,
                schedule=self.schedule,
                is_schedule_active=self.is_schedule_active,
                parameters=self.parameters,
                description=self.description,
                tags=self.tags,
                manifest_path=self.manifest_path,  # allows for backwards YAML compat
                path=self.path,
                entrypoint=self.entrypoint,
                infra_overrides=self.infra_overrides,
                storage_document_id=storage_document_id,
                infrastructure_document_id=infrastructure_document_id,
                parameter_openapi_schema=self.parameter_openapi_schema.dict(),
                pull_steps=pull_steps,
            )

            if client.server_type == ServerType.CLOUD:
                # The triggers defined in the deployment spec are, essentially,
                # anonymous and attempting truly sync them with cloud is not
                # feasible. Instead, we remove all automations that are owned
                # by the deployment, meaning that they were created via this
                # mechanism below, and then recreate them.
                await client.delete_resource_owned_automations(
                    f"prefect.deployment.{deployment_id}"
                )
                for trigger in self.triggers:
                    trigger.set_deployment_id(deployment_id)
                    await client.create_automation(trigger.as_automation())

            return deployment_id
